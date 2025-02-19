import argparse
import os

import tensorflow as tf
tf.enable_eager_execution()

# This import fails on the first try, but works on the second try. I don't know why.
try:
    import tensorflow_datasets as tfds
except:
    import tensorflow_datasets as tfds

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_data_dir', type=str, required=True)
    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    splits = ['mcd1', 'mcd2', 'mcd3']
    for split in splits:
        split_data_dir = os.path.join(args.root_data_dir, split)
        os.makedirs(split_data_dir, exist_ok=True)
        print(f'Split data dir: {split_data_dir}')
        train_file = open(os.path.join(split_data_dir, f'tasks_train_{split}.txt'), 'w')
        test_file = open(os.path.join(split_data_dir, f'tasks_test_{split}.txt'), 'w')
        data = tfds.load(f'scan/{split}')
        for i, sample in enumerate(data['train'].take(20000)):
            train_file.write(f'IN: {sample["commands"].numpy().decode()} OUT: {sample["actions"].numpy().decode()}\n')
        for i, sample in enumerate(data['test'].take(20000)):
            test_file.write(f'IN: {sample["commands"].numpy().decode()} OUT: {sample["actions"].numpy().decode()}\n')


if __name__ == '__main__':
    main()
