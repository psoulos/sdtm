import argparse
import os
import json
import shutil

from nltk.tree import Tree


def chomsky_normal_form(
    tree, factor="right", horzMarkov=None, vertMarkov=0, childChar="|", parentChar="|"
):
    # assume all subtrees have homogeneous children
    # assume all terminals have no siblings

    # A semi-hack to have elegant looking code below.  As a result,
    # any subtree with a branching factor greater than 999 will be incorrectly truncated.
    if horzMarkov is None:
        horzMarkov = 999

    # Traverse the tree depth-first keeping a list of ancestor nodes to the root.
    # I chose not to use the tree.treepositions() method since it requires
    # two traversals of the tree (one to get the positions, one to iterate
    # over them) and node access time is proportional to the height of the node.
    # This method is 7x faster which helps when parsing 40,000 sentences.

    nodeList = [(tree, [tree.label()])]
    while nodeList != []:
        node, parent = nodeList.pop()
        if isinstance(node, Tree):

            # parent annotation
            parentString = ""
            originalNode = node.label()
            if vertMarkov != 0 and node != tree and isinstance(node[0], Tree):
                parentString = "{}<{}>".format(parentChar, "-".join(parent))
                node.set_label(node.label() + parentString)
                parent = [originalNode] + parent[: vertMarkov - 1]

            # add children to the agenda before we mess with them
            for child in node:
                nodeList.append((child, parent))

            # chomsky normal form factorization
            if len(node) > 2:
                childNodes = []
                for child in node:
                    if isinstance(child, str):
                        childNodes.append(child)
                    else:
                        childNodes.append(child.label())
                #childNodes = [child.label() for child in node]
                nodeCopy = node.copy()
                node[0:] = []  # delete the children

                curNode = node
                numChildren = len(nodeCopy)
                for i in range(1, numChildren - 1):
                    if factor == "right":
                        newHead = "{}{}<{}>{}".format(
                            originalNode,
                            childChar,
                            "-".join(
                                childNodes[i : min([i + horzMarkov, numChildren])]
                            ),
                            parentString,
                        )  # create new head
                        newHead = f'{node.label()}1'
                        newNode = Tree(newHead, [])
                        curNode[0:] = [nodeCopy.pop(0), newNode]
                    else:
                        newHead = "{}{}<{}>{}".format(
                            originalNode,
                            childChar,
                            "-".join(
                                childNodes[max([numChildren - i - horzMarkov, 0]) : -i]
                            ),
                            parentString,
                        )
                        newHead = f'{node.label()}1'
                        newNode = Tree(newHead, [])
                        curNode[0:] = [newNode, nodeCopy.pop()]

                    curNode = newNode

                curNode[0:] = [child for child in nodeCopy]


def make_tree(output_str):
    root = Tree('answer', [])
    nodes = [root]
    current_parent = root
    prev_token = None
    outputs = output_str.split()[1:]
    for i, token in enumerate(outputs):
        if token == '(':
            pass
        elif token == ')':
            if len(current_parent) > 2:
                pass
                #print(current_parent)
                #print(output_str)
            current_parent = nodes.pop()
        elif token == ',':
            pass
        else:
            new_node = Tree(token, [])
            current_parent.append(new_node)
            if outputs[i+1] == '(':
                nodes.append(current_parent)
                current_parent = new_node
        prev_token = token
    assert len(nodes) == 0
    return root


def make_leaves_str(tree: Tree):
    """
    This method turns the leaf nodes in an nltk.tree.Tree into strings. This is important for printing out the
    tree as a string with parantheses.
    """
    if len(tree) == 0:
        return tree.label()
    else:
        new_root = Tree(tree.label(), [])
    for subtree in tree:
        new_root.append(make_leaves_str(subtree))

    return new_root


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--root_data_dir',
        type=str,
        required=True,
        help='The output of running the GeoQuery preprocessing from  '
             'https://github.com/google-research/language/tree/master/language/compgen/nqg#geoquery'
    )
    parser.add_argument('--version', type=int, required=True)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    output_dir = os.path.join(args.root_data_dir, f'v{str(args.version)}')
    os.makedirs(output_dir, exist_ok=False)
    splits = ['standard', 'length', 'template', 'tmcd']
    files = ['train.tsv', 'test.tsv']
    for split in splits:
        split_input_dir = os.path.join(args.root_data_dir, split)
        split_output_dir = os.path.join(output_dir, split)
        os.makedirs(split_output_dir, exist_ok=False)
        for file in files:
            output_filename = file.replace('tsv', 'json')
            with open(os.path.join(split_input_dir, file)) as f:
                data = f.readlines()
            with open(os.path.join(split_output_dir, output_filename), 'w') as f:
                depth_to_occurences = {}
                input_length_to_occurences = {}
                for line in data:
                    input_, output = line.split('\t')
                    output = output.strip()
                    output_tree = make_tree(output)
                    chomsky_normal_form(output_tree)
                    output_tree = make_leaves_str(output_tree)

                    height = output_tree.height()
                    if height not in depth_to_occurences:
                        depth_to_occurences[height] = 0
                    depth_to_occurences[height] += 1

                    input_length = len(input_.split())
                    if input_length not in input_length_to_occurences:
                        input_length_to_occurences[input_length] = 0
                    input_length_to_occurences[input_length] += 1

                    input_tree = ' '.join(map(lambda x: f'( {x} )', input_.split()))
                    output_tree = str(output_tree).replace('\n', '').replace('(', '( ').replace(')', ' )')
                    f.write(f'{json.dumps({"source": input_tree, "target": output_tree})}\n')
            with open(os.path.join(split_output_dir, output_filename.replace('.json', '_info.txt')), 'w') as f:
                f.write(f'Depths:\n')
                for depth, occurences in sorted(depth_to_occurences.items()):
                    f.write(f'{depth}\t{occurences}\n')
                f.write(f'Input lengths:\n')
                for input_length, occurences in sorted(input_length_to_occurences.items()):
                    f.write(f'{input_length}\t{occurences}\n')
        # Since these datasets don't have validation sets, copy the train set
        shutil.copy(os.path.join(split_output_dir, 'train.json'), os.path.join(split_output_dir, 'validation.json'))



if __name__ == '__main__':
    main()
