import argparse
import os
import json

from nltk.tree import Tree


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--root_data_dir',
        type=str,
        required=True,
        help='The location of an unzipped task from https://huggingface.co/datasets/rfernand/basic_sentence_transforms/blob/main/active_logical_ttb.zip'
    )
    parser.add_argument('--version', type=int, required=True)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    output_data_dir = os.path.join(args.root_data_dir, f'v{str(args.version)}')
    os.makedirs(output_data_dir, exist_ok=False)

    for file in os.listdir(args.root_data_dir):
        if not file.endswith('.jsonl'):
            continue
        print(f'Processing {file}')
        with open(os.path.join(args.root_data_dir, file), 'r') as f:
            lines = f.readlines()
        output_filename = file.replace('jsonl', 'json')
        with open(os.path.join(output_data_dir, output_filename), 'w') as f:
            for line in lines:
                inout_pair = json.loads(line)
                input_ = inout_pair['source']
                input_tree = Tree.fromstring(input_)
                input_tree = ' '.join(map(lambda x: f'( {x} )', input_tree.leaves()))
                f.write(f'{json.dumps({"source": input_tree, "target": inout_pair["target"]})}\n')

if __name__ == '__main__':
    main()
