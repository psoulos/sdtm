import argparse
import os
import json

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
                        newHead = '<NT>'
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
                        newHead = '<NT>'
                        newNode = Tree(newHead, [])
                        curNode[0:] = [newNode, nodeCopy.pop()]

                    curNode = newNode

                curNode[0:] = [child for child in nodeCopy]


def left_child_right_sibling(root: Tree):
    #print(root)
    if len(root) == 0 or isinstance(root, str):
        return root
    lcrs_subtrees = []
    for i, subtree in enumerate(root):
        lcrs_subtree = left_child_right_sibling(subtree)
        # If a terminal node that is a string has siblings, we need to convert it to a tree
        if i < len(root) - 1 and isinstance(lcrs_subtree, str):
            lcrs_subtree = Tree(lcrs_subtree, [])
        lcrs_subtrees.append(lcrs_subtree)
    current_sibling = lcrs_subtrees[0]
    for sibling in lcrs_subtrees[1:]:
        # If there are no siblings, add an EOB token
        if len(current_sibling) == 0:
            current_sibling.append('<EOB>')
        current_sibling.append(sibling)
        current_sibling = sibling
    lcrs_root = Tree(root.label(), [lcrs_subtrees[0]])
    return lcrs_root


def leafify(root: Tree):
    """Turns labeled non-terminal nodes into leaves and replaces with <NT>."""
    if len(root) == 0 or isinstance(root, str):
        return root
    new_subtrees = []
    for subtree in root:
        new_subtrees.append(leafify(subtree))
    new_root = Tree('<NT>', [])
    new_root.append(root.label())
    #root.set_label('<NT>')
    new_root.append(Tree('<NT>', new_subtrees))
    return new_root


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--root_data_dir',
        type=str,
        required=True,
        help='The location of wasp-2.0alpha/ from https://www.cs.utexas.edu/~ml/wasp/'
    )
    parser.add_argument('--version', type=int, required=True)
    parser.add_argument(
        '--no_labeled_nt',
        type=int,
        default=0,
        help='If set to 1, any labeled non-terminal nodes will be pushed to the left child and the labeled node will '
             'be replaced by <NT>. For example (A B C) will become (<NT> A (<NT> B C)).'
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    root_data_dir = os.path.join(args.root_data_dir, f'v{str(args.version)}')
    os.makedirs(root_data_dir, exist_ok=False)

    files = ['train.txt', 'test.txt', 'validation.txt']
    binarize_functions = ['lcrs', 'chomsky_normal']

    for binarize_function in binarize_functions:
        output_dir = os.path.join(root_data_dir, binarize_function)
        os.makedirs(output_dir, exist_ok=True)
        for file in files:
            with open(os.path.join(args.root_data_dir, file)) as f:
                data = f.readlines()
            output_filename = file.replace('txt', 'json')
            with open(os.path.join(output_dir, output_filename), 'w') as f:
                depth_to_occurences = {}
                for line in data:
                    input_, output = line.split('\t')
                    output = output.strip()

                    tree = Tree.fromstring(f'( {output} )')
                    if binarize_function == 'lcrs':
                        tree = left_child_right_sibling(tree)
                    elif binarize_function == 'chomsky_normal':
                        chomsky_normal_form(tree)

                    if args.no_labeled_nt:
                        tree = leafify(tree)

                    height = tree.height()
                    if height not in depth_to_occurences:
                        depth_to_occurences[height] = 0
                    depth_to_occurences[height] += 1

                    input_tree = ' '.join(map(lambda x: f'( {x} )', input_.split()))
                    output_tree = str(tree).replace('\n', '').replace('(', '( ').replace(')', ' )')
                    f.write(f'{json.dumps({"source": input_tree, "target": output_tree})}\n')
            with open(os.path.join(output_dir, file.replace('.txt', '_info.txt')), 'w') as f:
                for depth, occurences in sorted(depth_to_occurences.items()):
                    f.write(f'{depth}\t{occurences}\n')


if __name__ == '__main__':
    main()
