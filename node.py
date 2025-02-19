from enum import Enum

class Node:
    def __init__(self, label, children=None, key=None):
        self.label = label
        self.children = children if children else []

    def get_max_depth(self):
        max_child_depth = 0

        for child in self.children:
            max_child_depth = max(max_child_depth, child.get_max_depth())

        return 1 + max_child_depth

    def get_max_branching(self):
        max_branching = 1

        if self.children:
            max_branching = len(self.children)

            for child in self.children:
                max_branching = max(max_branching, child.get_max_branching())

        return max_branching

    def get_num_nodes(self):
        node_count = 1

        for child in self.children:
            node_count += child.get_num_nodes()

        return node_count
    
    def __repr__(self, max_len=None, depth=0, add_parens=True):

        label_key = self.label

        if self.children:
            child_rep = ""

            for c, child in enumerate(self.children):

                flat_child = len(child.children) == 0
                child_text = child.__repr__(depth=depth+1, add_parens=(not flat_child))

                if max_len and depth==0 and len(child_text) > max_len:
                    child_text = child_text[0:max_len] + "..."
                if c:
                    child_rep += " " + child_text
                else:
                    child_rep += child_text
            rep = "{} {}".format(label_key, child_rep)

        else:
            # simple node without children
            rep = label_key

        if add_parens:
            rep = "( {} )".format(rep)

        return rep
    
    def __str__(self):
        return self.__repr__()

    def str(self, max_len=0):
        return self.__repr__(max_len)


def left_child(index):
    # We encode the left child index by inserting a 0 between the most significant bit (msb) and the rest
    # of the bits. bin(int) always starts with '0b', so we start at index 3 to remove 0b and the msb.
    return int('10' + bin(index)[3:], 2)


def right_child(index):
    # We encode the right child index by inserting a 1 between the most significant bit (msb) and the rest
    # of the bits. bin(int) always starts with '0b', so we start at index 3 to remove 0b and the msb.
    return int('11' + bin(index)[3:], 2)


def root_node_index():
    return 1

class AddEobTokens(Enum):
    """
    Enum for whether to add EOB tokens to the tree.

    @ALL: add EOB tokens to all nodes
    @UNARY: add EOB tokens to unary nodes only
    @NONE: do not add EOB tokens to any nodes
    """
    ALL = 0
    UNARY = 1
    NONE = 2


def text_tree_to_node(tree, add_eob_tokens: AddEobTokens, add_eob_to_memory=False, num_extra_tokens_in_memory=0):
    '''
    text tree conventions:
        ( A ) ==> parent A
        ( A B ) ==> parent A with child B
        ( A B C ) ==> parent A with children B and C
        ( A ( B C ) ) ==> parent A with child B (B has child C)
    '''
    # If we receive a regular string, we wrap each word in parentheses to simulate root nodes
    if '(' not in tree:
        tree = tree.split()
        wrapped_words = ['( ' + word + ' )' for word in tree]
        tree = ' '.join(wrapped_words)

    completed_trees = []
    if add_eob_to_memory:
        eob_tree = Node('<EOB>')
        if add_eob_tokens == AddEobTokens.ALL:
            eob_tree.children.append(Node('<PAD>'))
            eob_tree.children.append(Node('<PAD>'))
        completed_trees.append(eob_tree)
    if num_extra_tokens_in_memory > 0:
        for i in range(num_extra_tokens_in_memory):
            # TODO: I think that these should be <EOB> tokens and not <PAD>?
            extra_token = Node(f'<TOKEN_{i}>')
            if add_eob_tokens == AddEobTokens.ALL:
                extra_token.children.append(Node('<EOB>'))
                extra_token.children.append(Node('<EOB>'))
            completed_trees.append(extra_token)
    prev_tok = None
    stack = []
    parent = None
    for tok in tree.split():
        if prev_tok == '(':
            label = tok
            parent = Node(label)
            stack.append(parent) 
        elif tok == ')':
            child = parent
            if len(child.children) == 0 and add_eob_tokens == AddEobTokens.ALL:
                child.children.append(Node('<EOB>'))
                child.children.append(Node('<EOB>'))
            elif len(child.children) == 1 and (add_eob_tokens == AddEobTokens.UNARY or add_eob_tokens ==
                                               AddEobTokens.ALL):
                child.children.append(Node('<EOB>'))
            stack.pop()
            if len(stack) > 0:
                parent = stack[-1]
                parent.children.append(child)
            elif len(stack) == 0:
                completed_trees.append(parent)
        elif tok != '(':
            label = tok
            node = Node(label)
            if label != '<EOB>' and add_eob_tokens == AddEobTokens.ALL:
                node.children.append(Node('<EOB>'))
                node.children.append(Node('<EOB>'))
            parent.children.append(node)
        prev_tok = tok

    return completed_trees
