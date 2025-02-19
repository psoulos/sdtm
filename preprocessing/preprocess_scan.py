import argparse
import os
import random
import json
import shutil

import math
import nltk


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_data_dir', type=str, required=True)
    parser.add_argument('--split', type=str, required=True)
    parser.add_argument('--jump_percentage', type=float)
    parser.add_argument('--version', type=int, required=True)
    parser.add_argument('--validation_percentage', type=float, default=.25)
    parser.add_argument(
        '--push_to_depth',
        type=int,
        default=None,
        help='If not None, the sequence of tokens will be pushed to the given depth. For example, if set to 3, '
             'the output for "jump" would be [ ( <NT> ( <NT> ( <NT> jump <EOB> ) <EOB> ) <EOB> ) ]'
    )
    parser.add_argument(
        '--push_single_output_node_down',
        type=int,
        default=0,
        help='If 1 and the output has length 1 [ie (JUMP) ] push the single output node down to the left child '
             '[ie ( <NT> JUMP ) ]'
    )
    args = parser.parse_args()

    if args.split == 'add_prim_split/jump' and args.jump_percentage is None:
        raise ValueError('Must provide jump percentage for add_prim_split/jump')

    assert args.push_single_output_node_down in [0, 1]
    return args


sample_input_key = 'source'
sample_output_key = 'target'


def convert_raw_sample_to_input_output_dict(sample):
    input_, output = sample.strip().split('OUT: ')
    input_ = input_.split('IN: ')[1].strip()
    return {sample_input_key: input_, sample_output_key: output}


def build_separate_tree(strings):
    return ' '.join(map(lambda x: f'( {x} )', strings))


def build_leaves_tree(strings, push_single_output_node_down=False, push_to_depth=None):
    if len(strings) == 1:
        if push_single_output_node_down:
            return f'( <NT> {strings[0]} )'
        else:
            return f'( {strings[0]} )'
    """
    This function builds a tree with the given strings as leaves.
    """
    def create_tree(level, max_level):
        if level == max_level:
            return strings.pop(0) if strings else None
        else:
            left = create_tree(level + 1, max_level)
            right = create_tree(level + 1, max_level)
            if left is None:
                return
            elif right is None:
                return f'( <NT> {left} )'
            else:
                return f'( <NT> {left} {right} )'
    # Determine the depth of the tree based on the number of strings
    depth = math.ceil(math.log2(len(strings)))
    if push_to_depth and depth > push_to_depth:
        raise ValueError(f'Cannot push to depth {push_to_depth} because the depth of the current tree is {depth}: '
                         f'{strings}')
    if push_to_depth:
        depth = push_to_depth
    return create_tree(0, depth)


def build_in_order_tree(strings, depth=0):
    """
    This function builds a tree with the given strings using in-order traversal.
    """
    if len(strings) == 1 and depth == 0:
        return f'( {strings[0]} )'

    if not strings:
        return None

    mid = len(strings) // 2
    root = strings[mid]

    left = build_in_order_tree(strings[:mid], depth=depth+1)
    right = build_in_order_tree(strings[mid + 1:], depth=depth+1)

    output = None
    if root is None:
        output = None
    elif left is None:
        output = f'{root}'
    elif right is None:
        output = f'( {root} {left} )'
    else:
        output = f'( {root} {left} {right} )'

    return output


def build_sequitor_tree(strings, parser):
    f = open('../SCAN/tasks.txt', 'r')
    dataset = f.readlines()

    from sksequitur import Grammar, Parser, Mark
    parser = Parser()
    for line in dataset:
        tokens = line.split('OUT: ')[1].strip().split()
        for token in tokens:
            if token == 'I_TURN_LEFT':
                parser.feed('E')
            elif token == 'I_TURN_RIGHT':
                parser.feed('I')
            elif token == 'I_JUMP':
                parser.feed('J')
            elif token == 'I_LOOK':
                parser.feed('L')
            elif token == 'I_RUN':
                parser.feed('R')
            elif token == 'I_WALK':
                parser.feed('W')
            else:
                raise ValueError(f'Unknown string {token}')
        parser.feed([Mark()])
    grammar = Grammar(parser.tree)

    dataset_index_to_productions = {}
    offset = 0
    mark_type = type(Mark())
    for i in range(len(dataset)):
        sample_i_productions = []
        while offset < len(grammar[0]) and type(grammar[0][offset]) != mark_type:
            sample_i_productions.append(grammar[0][offset])
            offset += 1
        dataset_index_to_productions[i] = sample_i_productions
        # Move past the Mark
        offset += 1

    # TODO: how do I deal with non-binary production rules?

    print(grammar)

    return parser
    grammar = Grammar(parser.tree)


def build_tree_from_grammar(grammar, dataset_index):
    # The odd indices are the Mark separators between samples
    root_production = grammar[0][dataset_index*2]
    if root_production in ['E', 'I', 'J', 'L', 'R', 'W']:
        return f'( <NT> {root_production} )'

    def build_tree_from_productions(grammar, production_index):
        production0, production1 = grammar[production_index]
        if production0 in ['E', 'I', 'J', 'L', 'R', 'W']:
            result0 = f'{production0}'
        elif production0:
            result0 = f'{build_tree_from_productions(grammar, production0)}'
        else:
            result0 = ''

        if production1 in ['E', 'I', 'J', 'L', 'R', 'W']:

            result1 = f'{production1}'
        elif production1:
            result1 = f'{build_tree_from_productions(grammar, production1)}'
        else:
            result1 = ''

        return f'( <NT> {result0} {result1} )'

    return f'{build_tree_from_productions(grammar, root_production)}'


def build_pre_order_tree(strings, depth=0):
    """
    This function builds a tree with the given strings using in-order traversal.
    """
    if len(strings) == 1 and depth == 0:
        return f'( {strings[0]} )'

    if not strings:
        return None

    mid = len(strings) // 2
    root = strings[0]

    left = build_pre_order_tree(strings[1:mid + 1], depth=depth+1)
    right = build_pre_order_tree(strings[mid + 1:], depth=depth+1)

    output = None
    if root is None:
        output = None
    elif left is None:
        output = f'{root}'
    elif right is None:
        output = f'( {root} {left} )'
    else:
        output = f'( {root} {left} {right} )'

    return output


def build_left_branching_tree(strings):
    if not strings:
        return '<EOB>'

    return f'( <PAD> {build_left_branching_tree(strings[:-1])} {strings[-1]} )'


def build_right_branching_tree(strings):
    if not strings:
        return ''

    return f'( <PAD> {strings[0]} {build_right_branching_tree(strings[1:])} )'


def max_depth_(tree_string):
    # If the string is ( A ), then the depth is 1
    if len(tree_string.split()) == 3:
        return 1
    current_depth = 1
    max_depth = 1

    for char in tree_string:
        if char == '(':
            current_depth += 1
            max_depth = max(max_depth, current_depth)
        elif char == ')':
            current_depth -= 1

    return max_depth


def change_non_terminals_to_NT(tree):
    """
    Recursively change all non-terminal labels in the tree to "<NT>".
    """
    # Base case: If the tree is a leaf, just return it without modification
    if isinstance(tree, str):
        return tree
    # Recursive case: The tree is not a leaf
    else:
        # Change the current node label to "<NT>"
        tree.set_label("<NT>")
        # Recursively apply the function to all subtrees (children)
        for idx, subtree in enumerate(tree):
            change_non_terminals_to_NT(subtree)
        return tree

scan_grammar = nltk.CFG.fromstring("""
C -> S CONJ S
C -> S CONJ S
C -> S
S -> V 'twice'
S -> V 'thrice'
S -> V
V -> D1 M D2
D1 -> U
D1 -> 'turn'
D2 -> 'left'
D2 -> 'right'
V -> D
V -> U
D -> U 'left'
D -> U 'right'
D -> 'turn' 'left'
D -> 'turn' 'right'
U -> 'walk'
U -> 'look'
U -> 'run'
U -> 'jump'

M -> 'opposite'
M -> 'around'
CONJ -> 'and'
CONJ -> 'after'
""")

parser = nltk.ChartParser(scan_grammar)

def build_in_parsed_tree(strings):
    tree = list(parser.parse(strings))[0]
    nltk.chomsky_normal_form(tree)
    change_non_terminals_to_NT(tree)
    tree_string = tree.pformat()
    return tree_string.replace('\n ', '').replace('(', '( ').replace(')', ' )')


def build_out_parsed_tree(strings, input_to_output):
    subseq1 = None
    subseq2 = None
    if 'and' in strings:
        subseq1, subseq2 = strings.split('and')
        return (f'( <NT> {build_out_parsed_tree(subseq1.strip(), input_to_output)} '
                f'{build_out_parsed_tree(subseq2.strip(), input_to_output)} )')
    elif 'after' in strings:
        # Note that subseq1 and subseq2 are flipped
        subseq2, subseq1 = strings.split('after')
        return (f'( <NT> {build_out_parsed_tree(subseq1.strip(), input_to_output)} '
                f'{build_out_parsed_tree(subseq2.strip(), input_to_output)} )')

    if 'thrice' in strings:
        subseq = strings.split('thrice')[0].strip()
        parsed_subseq = build_out_parsed_tree(subseq, input_to_output)
        return f'( <NT> ( <NT> {parsed_subseq} {parsed_subseq} ) {parsed_subseq} )'
    elif 'twice' in strings:
        subseq = strings.split('twice')[0].strip()
        parsed_subseq = build_out_parsed_tree(subseq, input_to_output)
        return f'( <NT> {parsed_subseq} {parsed_subseq} )'

    if 'around' in strings:
        action, direction = strings.split(' around ')
        if action == 'turn':
            return f'( <NT> ( <NT> {input_to_output[direction]} {input_to_output[direction]} ) ( <NT> {input_to_output[direction]} {input_to_output[direction]} ) )'
        else:
            return (
                f'( <NT> ( <NT> ( <NT> {input_to_output[direction]} {input_to_output[action]} ) ( <NT> '
                f'{input_to_output[direction]} {input_to_output[action]} ) ) ( <NT> ( <NT> '
                f'{input_to_output[direction]} {input_to_output[action]} ) ( <NT> {input_to_output[direction]} '
                f'{input_to_output[action]} ) ) )'
            )
    elif 'opposite' in strings:
        action, direction = strings.split(' opposite ')
        if action == 'turn':
            return f'( <NT> {input_to_output[direction]} {input_to_output[direction]} )'
        else:
            return (f'( <NT> ( <NT> {input_to_output[direction]} {input_to_output[direction]} )'
                    f' {input_to_output[action]} )')

    if len(strings.split()) == 2:
        # action direction
        action, direction = strings.split()
        if action == 'turn':
            return f'{input_to_output[direction]}'
        return f'( <NT> {input_to_output[direction]} {input_to_output[action]} )'
    else:
        # just action
        return f'{input_to_output[strings]}'

def build_parsed_tree(strings, push_single_output_node_down=False, lowercase=False):
    in_tree = build_in_parsed_tree(strings.split())
    if lowercase:
        input_to_output = {
            'left': 'left',
            'right': 'right',
            'walk': 'walk',
            'look': 'look',
            'run': 'run',
            'jump': 'jump',
        }
    else:
        input_to_output = {
            'left': 'I_TURN_LEFT',
            'right': 'I_TURN_RIGHT',
            'walk': 'I_WALK',
            'look': 'I_LOOK',
            'run': 'I_RUN',
            'jump': 'I_JUMP',
        }
    if len(strings.split()) == 1 and push_single_output_node_down:
        out_tree = f'( <NT> {input_to_output[strings]} )'
    else:
        out_tree = build_out_parsed_tree(strings, input_to_output)
    return in_tree, out_tree


def main():
    args = parse_args()
    split_data_dir = os.path.join(args.root_data_dir, args.split)
    print(f'Split data dir: {split_data_dir}')
    seed = random.randint(0, 65535)
    print(f'Seed: {seed}')
    random.seed(seed)
    contains_dev = False
    for file in os.listdir(split_data_dir):
        # If we find a dev file, we will not extract a validation set from the training data
        if 'dev' in file:
            args.validation_percentage = 0
    for file in os.listdir(split_data_dir):
        if 'train' in file:
            train_samples = open(os.path.join(split_data_dir, file)).readlines()

            # Remove all jump samples, we will add them back later
            if args.split == 'add_prim_split/jump':
                print(f'Jump percentage: {args.jump_percentage}%')
                # Remove all jump samples
                jump_sample = 'IN: jump OUT: I_JUMP\n'
                n_jump_samples = 0
                _train_samples = []
                for sample in train_samples:
                    if sample == jump_sample:
                        n_jump_samples += 1
                    else:
                        _train_samples.append(sample)
                train_samples = _train_samples

            validation_percentage = args.validation_percentage
            sampled_indices = random.sample(range(len(train_samples)), int(len(train_samples) * validation_percentage))
            validation_samples = []
            if not contains_dev and validation_percentage > 0:
                validation_samples = [train_samples[i] for i in sampled_indices]

                # Sort indices in descending order so that removing an element doesn't shift the indices of the remaining
                # elements to remove
                sampled_indices.sort(reverse=True)
                for sample_index in sampled_indices:
                    del train_samples[sample_index]

            # Add back the jump samples
            if args.split == 'add_prim_split/jump':
                # By default, jump makes up 10% of the training data. Adjust this percentage by the jump_percentage.
                n_jump_samples = int(n_jump_samples * args.jump_percentage / 100)
                for _ in range(int(n_jump_samples * (1 - validation_percentage))):
                    train_samples.append(jump_sample)
                for _ in range(int(n_jump_samples * validation_percentage)):
                    validation_samples.append(jump_sample)

        elif 'test' in file:
            test_samples = open(os.path.join(split_data_dir, file)).readlines()
        elif 'dev' in file:
            validation_samples = open(os.path.join(split_data_dir, file)).readlines()

    train_samples = list(map(convert_raw_sample_to_input_output_dict, train_samples))
    validation_samples = list(map(convert_raw_sample_to_input_output_dict, validation_samples))
    test_samples = list(map(convert_raw_sample_to_input_output_dict, test_samples))

    tree_splits = [
        'separate_to_leaves',
        'separate_to_parsed',
        #'parsed_to_parsed',
        #'parsed_to_leaves',
        #'separate_to_in_order',
        #'separate_to_pre_order',
        #'separate_to_left_branching',
        #'leaves_to_leaves',
        #'in_order_to_in_order',
        #'pre_order_to_pre_order',
        #'left_branching_to_left_branching',
        #'right_branching_to_right_branching',
    ]
    for tree_split in tree_splits:
        print(f'Tree split {tree_split}')
        new_split_data_dir = os.path.join(split_data_dir, f'v{args.version}', f'{tree_split}')
        os.makedirs(new_split_data_dir, exist_ok=False)
        input_type, output_type = tree_split.split('_to_')
        input_transformation_fn = globals()[f'build_{input_type}_tree']
        output_transformation_fn = globals()[f'build_{output_type}_tree']

        for name, data_split in zip(['train', 'validation', 'test'], [train_samples, validation_samples, test_samples]):
            print(f'Split name {name}')
            max_tree_depth = 0
            input_trees = []
            output_trees = []
            for i, sample in enumerate(data_split):
                # Input Tree
                in_out_tuple = None
                if input_transformation_fn == build_parsed_tree:
                    in_out_tuple = input_transformation_fn(sample['source'])
                    input_tree = in_out_tuple[0]
                else:
                    input_tree = input_transformation_fn(sample['source'].split())
                input_trees.append(input_tree)

                # Output Tree
                if output_transformation_fn == build_parsed_tree:
                    if in_out_tuple:
                        output_tree = in_out_tuple[1]
                    else:
                        output_tree = output_transformation_fn(sample['source'],
                            args.push_single_output_node_down, lowercase=args.split=='zero_split')[1]
                elif output_transformation_fn == build_leaves_tree:
                    output_tree = output_transformation_fn(sample['target'].split(),
                        args.push_single_output_node_down, args.push_to_depth)
                else:
                    output_tree = output_transformation_fn(sample['target'].split())
                output_trees.append(output_tree)
                # print(f'Sample {i}: input depth {max_depth_(input_tree)}, output depth {max_depth_(output_tree)}')
                # The depth is off by one because (A B C) is depth 1, not 2 since we only count ()
                max_tree_depth = max(max_tree_depth, max_depth_(output_tree) + 1, max_depth_(input_tree) + 1)

            with open(os.path.join(new_split_data_dir, f'{name}.json'), 'w') as f:
                for input_tree, output_tree in zip(input_trees, output_trees):
                    f.write(f'{json.dumps({"source": input_tree, "target": output_tree})}\n')
            with open(os.path.join(new_split_data_dir, f'{name}_info.txt'), 'w') as f:
                f.write('Max tree depth: {}\n'.format(max_tree_depth))
        if len(validation_samples) == 0:
            # If this split doesn't have validation sets, copy the train set
            shutil.copy(os.path.join(new_split_data_dir, 'train.json'), os.path.join(new_split_data_dir, 'validation.json'))


if __name__ == '__main__':
    main()
