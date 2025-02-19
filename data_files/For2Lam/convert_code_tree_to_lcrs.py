import json
from nltk.tree import Tree

from node import *


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

class BinaryNode:
    def __init__(self, label):
        self.label = label
        self.left = None
        self.right = None

    def __repr__(self, max_len=None, add_parens=True):

        label_key = self.label

        if self.left or self.right:
            child_rep = ""

            if self.right and self.left:
                child_rep += self.left.__repr__(add_parens=self.left.left or self.left.right) + " " \
                             + self.right.__repr__(add_parens=self.right.left or self.right.right)
            elif self.left:
                child_rep += self.left.__repr__(add_parens=self.left.left or self.left.right)
            elif self.right:
                child_rep += "\u1f600 " + self.right.__repr__(add_parens=self.right.left or self.right.right)

            rep = "{} {}".format(label_key, child_rep)
        else:
            # simple node without children
            rep = label_key

        if add_parens:
            rep = "( {} )".format(rep)

        return rep

    def __str__(self):
        return self.__repr__()


def convert_to_lcrs(root):
    new_root = BinaryNode(root.label)

    if len(root.children) > 0:
        new_root.left = convert_to_lcrs(root.children[0])
        for sibling in reversed(root.children[1:]):
            new_child = convert_to_lcrs(sibling)
            new_child.right = new_root.left.right
            new_root.left.right = new_child
    return new_root


for data_dir in ['small']:#, 'large']:
    print(f'Data dir: {data_dir}')
    for data_filename in ['train', 'test', 'valid']:
        print(f'File: {data_filename}')
        with open(f'data_files/For2Lam/{data_dir}/progs_{data_filename}.json', 'r') as f:
            data = json.load(f)

        with open(f'data_files/For2Lam/{data_dir}/progs_{data_filename}_lcrs.xy', 'w') as f:
            for d in data:
                x = d['for_tree']
                y = d['lam_tree']
                x_tree = fromlist(x)
                y_tree = fromlist(y)

                x_tree_str = ' '.join(str(x_tree).split()).replace("'", "").replace('(', '( ').replace(')', ' )')
                y_tree_str = ' '.join(str(y_tree).split()).replace("'", "").replace('(', '( ').replace(')', ' )')

                x_node_tree = text_tree_to_node(x_tree_str)
                y_node_tree = text_tree_to_node(y_tree_str)

                lcrs_x_tree = convert_to_lcrs(x_node_tree)
                lcrs_y_tree = convert_to_lcrs(y_node_tree)

                f.write(f'{lcrs_x_tree}\t{lcrs_y_tree}\n')
