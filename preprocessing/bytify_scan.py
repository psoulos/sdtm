import argparse
import os
import json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--split_data_dir',
        type=str,
        required=True,
    )
    args = parser.parse_args()
    return args


output_word_to_byte = {
    'I_TURN_LEFT': 'E',
    'I_TURN_RIGHT': 'I',
    'I_JUMP': 'J',
    'I_LOOK': 'L',
    'I_RUN': 'R',
    'I_WALK': 'W',
}
output_byte_to_word = {v: k for k, v in output_word_to_byte.items()}


def main():
    args = parse_args()
    output_data_dir = args.split_data_dir

    for file in os.listdir(args.split_data_dir):
        if not file.endswith('.txt'):
            continue
        print(f'Processing {file}')
        with open(os.path.join(args.split_data_dir, file), 'r') as f:
            lines = f.readlines()
        output_filename = file.replace('txt', 'out.byte.txt')
        with open(os.path.join(output_data_dir, output_filename), 'w') as f:
            for line in lines:
                tokens = line.split('OUT: ')[1].strip().split()
                byte_string = list(map(lambda x: output_word_to_byte[x], tokens))
                f.write(f'{"".join(byte_string)}\n')

if __name__ == '__main__':
    main()
