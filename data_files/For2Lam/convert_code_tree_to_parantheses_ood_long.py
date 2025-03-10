import json
import os
from nltk.tree import Tree


def fromlist(l):
    """
    :type l: list
    :param l: a tree represented as nested lists

    :return: A tree corresponding to the list representation ``l``.
    :rtype: Tree

    Convert nested lists to a NLTK Tree
    """
    if type(l) is not list:
        return l
    if type(l) == list and len(l) > 0:
        label = repr(l[0])
        if len(l) > 1:
            return Tree(label, [fromlist(child) for child in l[1:]])
        else:
            return label


def chomsky_normal_form(
    tree, factor="right", horzMarkov=None, vertMarkov=0, childChar="|", parentChar="^"
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
                    if isinstance(child, str) or isinstance(child, int):
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
                        newNode = Tree(newHead, [])
                        curNode[0:] = [newNode, nodeCopy.pop()]

                    curNode = newNode

                curNode[0:] = [child for child in nodeCopy]


def alter_binarized_node_labels(tree, count=-1, remove_new_nonterminals=False):
    '''
    Traverse the tree to alter the binary branching childChar parts
    If count==-1, just remove the childChar text. If count>=0, replace the childChar text with the count
    '''
    if '|' in tree.label():
        if count == -1:
            tree.set_label(tree.label()[:tree.label().index('|')])
        else:
            if remove_new_nonterminals:
                tree.set_label('\u2205')
            else:
                tree.set_label(tree.label()[:tree.label().index('|')+1] + str(count))
            count += 1
    else:
        if count != -1:
            count = 0

    for subtree in tree:
        if type(subtree) == Tree:
            alter_binarized_node_labels(subtree, count, remove_new_nonterminals)

version = 2
long_depths = [8, 10]
for feature in ['with_nt', 'without_nt']:
    print(f'Feature: {feature}')
    for data_dir in ['small']:#, 'large']:
        print(f'Data dir: {data_dir}')
        depth_to_data_dir = {}
        depth_to_long_file = {}
        depth_to_shorter_samples = {}
        for long_depth in long_depths:
            ood_long_data_dir = f'v{version}/{data_dir}/ood_long_{long_depth}/{feature}'
            depth_to_data_dir[long_depth] = ood_long_data_dir
            os.makedirs(ood_long_data_dir, exist_ok=True)
            long_gen_file = open(os.path.join(ood_long_data_dir, 'ood_long.xy'), 'w')
            depth_to_long_file[long_depth] = long_gen_file
            depth_to_shorter_samples[long_depth] = []
        original_data_dir = f'v{version}/{data_dir}/original/{feature}'
        os.makedirs(original_data_dir, exist_ok=True)
        for data_filename in ['train', 'test', 'valid']:
            print(f'File: {data_filename}')
            with open(f'{data_dir}/progs_{data_filename}.json', 'r') as json_file:
                data = json.load(json_file)
                out_filename = data_filename if data_filename != 'valid' else 'dev'
                with open(os.path.join(original_data_dir, f'{out_filename}.xy'), 'w') as original_out_file:
                    original_lexical_gen_file = open(os.path.join(original_data_dir, 'ood_new.xy'), 'w') if data_filename == 'test' else None

                    for d in data:
                        x = d['for_tree']
                        y = d['lam_tree']
                        x_tree = fromlist(x)
                        y_tree = fromlist(y)

                        chomsky_normal_form(x_tree)
                        chomsky_normal_form(y_tree)

                        alter_binarized_node_labels(x_tree, 0, remove_new_nonterminals=feature == 'without_nt')
                        alter_binarized_node_labels(y_tree, 0, remove_new_nonterminals=feature == 'without_nt')

                        x_tree_str = ' '.join(str(x_tree).split()).replace("'", "").replace('(', '( ').replace(')', ' )')
                        y_tree_str = ' '.join(str(y_tree).split()).replace("'", "").replace('(', '( ').replace(')', ' )')
                        original_out_file.write(f'{x_tree_str}\t{y_tree_str}\n')

                        if original_lexical_gen_file:
                            ood_x_tree_str = x_tree_str.replace(' x ', ' z ')
                            ood_y_tree_str = y_tree_str.replace(' x ', ' z ')
                            original_lexical_gen_file.write(f'{ood_x_tree_str}\t{ood_y_tree_str}\n')

                        for long_depth in long_depths:
                            if x_tree.height() > long_depth:
                                depth_to_long_file[long_depth].write(f'{x_tree_str}\t{y_tree_str}\n')
                            else:
                                depth_to_shorter_samples[long_depth].append(f'{x_tree_str}\t{y_tree_str}\n')

        for long_depth in long_depths:
            print('Long depth: {}'.format(long_depth))
            shorter_samples = depth_to_shorter_samples[long_depth]
            print(f'Number of shorter samples: {len(shorter_samples)}')
            train_samples = shorter_samples[:int(len(shorter_samples)*0.8)]
            dev_samples = shorter_samples[int(len(shorter_samples)*0.8):int(len(shorter_samples)*0.9)]
            test_samples = shorter_samples[int(len(shorter_samples)*0.9):]
            ood_long_data_dir = depth_to_data_dir[long_depth]
            train_file = open(os.path.join(ood_long_data_dir, 'train.xy'), 'w')
            dev_file = open(os.path.join(ood_long_data_dir, 'dev.xy'), 'w')
            test_file = open(os.path.join(ood_long_data_dir, 'test.xy'), 'w')
            lexical_gen_file = open(os.path.join(ood_long_data_dir, 'ood_new.xy'), 'w')
            for sample in train_samples:
                train_file.write(sample)
            for sample in dev_samples:
                dev_file.write(sample)
            for sample in test_samples:
                test_file.write(sample)
                x_tree_str, y_tree_str = sample.split('\t')
                ood_x_tree_str = x_tree_str.replace(' x ', ' z ')
                ood_y_tree_str = y_tree_str.replace(' x ', ' z ')
                lexical_gen_file.write(f'{ood_x_tree_str}\t{ood_y_tree_str}')
