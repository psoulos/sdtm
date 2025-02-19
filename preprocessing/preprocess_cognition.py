import argparse
import os
import json

from nltk.tree import Tree

from preprocess_scan import build_leaves_tree


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--root_data_dir',
        type=str,
        required=True,
        help='The `cognition_cg` subdirectory from unzipping '
             'https://drive.google.com/file/d/1hh3zCmfObd_6E8rtPcPOsP3gLe73JR2G/view.'
    )
    parser.add_argument('--version', type=int, required=True)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    output_dir = os.path.join(args.root_data_dir, f'v{str(args.version)}')
    os.makedirs(output_dir, exist_ok=False)
    files = ['test', 'train', 'valid']
    for file in files:
        en_file = f'{file}.en-zh.en'
        zh_file = f'{file}.en-zh.zh'
        output_file = f'{file}.json' if file != 'valid' else f'validation.json'

        en_data = open(os.path.join(args.root_data_dir, en_file), 'r').readlines()
        zh_data = open(os.path.join(args.root_data_dir, zh_file), 'r').readlines()

        max_input_len = 0
        max_output_depth = 0
        with open(os.path.join(output_dir, output_file), 'a') as f:
            for en_input, zh_output in zip(en_data, zh_data):
                input_tree = ' '.join(map(lambda x: f'( {x} )', en_input.strip().split()))
                output_tree = build_leaves_tree(zh_output.strip().split())
                f.write(f'{json.dumps({"source": input_tree, "target": output_tree})}\n')

                max_input_len = max(max_input_len, len(en_input.strip().split()))
                max_output_depth = max(max_output_depth, Tree.fromstring(output_tree).height())
        with open(os.path.join(output_dir, f'{file}_info.txt'), 'w') as f:
            f.write(f'Max input length: {max_input_len}\n')
            f.write(f'Max output depth: {max_output_depth}\n')


if __name__ == '__main__':
    main()
