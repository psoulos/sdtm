import json
import os
import random
import re
from typing import Dict, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, DistributedSampler

from node import text_tree_to_node, left_child, right_child, root_node_index


class Lang:
    def __init__(self, name, add_eob_tokens=True):
        self.name = name
        self.vocab2ind = {'<PAD>': 0}
        self.ind2vocab = {0: '<PAD>'}
        if add_eob_tokens:
            eob_idx = 1
            self.vocab2ind['<EOB>'] = eob_idx
            self.ind2vocab[eob_idx] = '<EOB>'

    def add_word(self, word):
        if word not in self.vocab2ind:
            self.ind2vocab[len(self.vocab2ind)] = word
            self.vocab2ind[word] = len(self.vocab2ind)


def prepare_data_loaders(data_dir: str, max_depth: int, add_eob_tokens: bool, is_ddp: bool, batch_size: int,
                         num_workers: int, data_filter: str = None, max_train_examples: int = None,
                         output_lowercase: bool = False, add_eob_to_memory: bool = False,
                         num_extra_tokens_in_memory: int = 0
                         ) -> Tuple[Dict[str, DataLoader], Lang, Lang]:
    data_filter = re.compile(data_filter) if data_filter is not None else None
    is_json_data = 'train.json' in os.listdir(data_dir)

    input_lang = Lang('input', add_eob_tokens=add_eob_tokens)
    output_lang = Lang('output', add_eob_tokens=add_eob_tokens)

    train_data = BinaryT2TDataset(
        os.path.join(data_dir, 'train.json' if is_json_data else 'train.xy'),
        input_lang,
        output_lang,
        max_depth=max_depth,
        filter_=data_filter if data_filter else None,
        max_examples=max_train_examples,
        add_eob_tokens=add_eob_tokens,
        output_lowercase=output_lowercase,
        add_eob_to_memory=add_eob_to_memory,
        num_extra_tokens_in_memory=num_extra_tokens_in_memory
    )
    print('{} training examples'.format(len(train_data)))
    # if is_ddp, we should use sampler=DistributedSampler(dataset) and set shuffle=False
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=not is_ddp,
                              num_workers=num_workers,
                              pin_memory=True, collate_fn=_sparse_collate_fn,
                              sampler=None if not is_ddp else DistributedSampler(train_data))

    valid_data = BinaryT2TDataset(
        os.path.join(data_dir, 'validation.json' if is_json_data else 'dev.xy'),
        input_lang,
        output_lang,
        max_depth=max_depth,
        filter_=data_filter if data_filter else None,
        add_eob_tokens=add_eob_tokens,
        output_lowercase=output_lowercase,
        add_eob_to_memory=add_eob_to_memory,
        num_extra_tokens_in_memory=num_extra_tokens_in_memory,
    )
    print('{} valid examples'.format(len(valid_data)))
    # Double the batch size for validation since we use less memory when not training
    valid_loader = DataLoader(valid_data, batch_size=batch_size*2, shuffle=True, num_workers=num_workers,
                              pin_memory=True, collate_fn=_sparse_collate_fn)

    test_data = BinaryT2TDataset(
        os.path.join(data_dir, 'test.json' if is_json_data else 'test.xy'),
        input_lang,
        output_lang,
        max_depth=max_depth,
        filter_=data_filter if data_filter else None,
        add_eob_tokens=add_eob_tokens,
        output_lowercase=output_lowercase,
        add_eob_to_memory=add_eob_to_memory,
        num_extra_tokens_in_memory=num_extra_tokens_in_memory,
    )
    print('{} test examples'.format(len(test_data)))
    # Double the batch size for validation since we use less memory when not training
    test_loader = DataLoader(test_data, batch_size=batch_size*2, shuffle=False, num_workers=num_workers,
                             pin_memory=True, collate_fn=_sparse_collate_fn)

    eval_long_loader = None
    eval_new_loader = None
    eval_illformed_loader = None
    if os.path.isfile(os.path.join(data_dir, 'ood_long.json' if is_json_data else 'ood_long.xy')):
        eval_long_data = BinaryT2TDataset(
            os.path.join(data_dir, 'ood_long.json' if is_json_data else 'ood_long.xy'),
            input_lang,
            output_lang,
            max_depth=max_depth,
            filter_=data_filter if data_filter else None,
            add_eob_tokens=add_eob_tokens,
            output_lowercase=output_lowercase,
            add_eob_to_memory=add_eob_to_memory,
            num_extra_tokens_in_memory=num_extra_tokens_in_memory,
        )
        eval_long_loader = DataLoader(eval_long_data, batch_size=batch_size, shuffle=False,
                                      num_workers=num_workers, pin_memory=True, collate_fn=_sparse_collate_fn)
        print('{} ood_long examples'.format(len(eval_long_data)))

    if os.path.isfile(os.path.join(data_dir, 'ood_new.json' if is_json_data else 'ood_new.xy')):
        eval_new_data = BinaryT2TDataset(
            os.path.join(data_dir, 'ood_new.json' if is_json_data else 'ood_new.xy'),
            input_lang,
            output_lang,
            max_depth=max_depth,
            filter_=data_filter if data_filter else None,
            add_eob_tokens=add_eob_tokens,
            output_lowercase=output_lowercase,
            add_eob_to_memory=add_eob_to_memory,
            num_extra_tokens_in_memory=num_extra_tokens_in_memory,
        )
        eval_new_loader = DataLoader(eval_new_data, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                     pin_memory=True, collate_fn=_sparse_collate_fn)
        print('{} ood_new examples'.format(len(eval_new_data)))

    if os.path.isfile(os.path.join(data_dir, 'ood_illformed.xy')):
        eval_illformed_data = BinaryT2TDataset(
            os.path.join(data_dir, 'ood_illformed.xy'),
            input_lang,
            output_lang,
            max_depth=max_depth,
            filter_=data_filter if data_filter else None,
            add_eob_tokens=add_eob_tokens,
            output_lowercase=output_lowercase,
            add_eob_to_memory=add_eob_to_memory,
            num_extra_tokens_in_memory=num_extra_tokens_in_memory,
        )
        eval_illformed_loader = DataLoader(eval_illformed_data, batch_size=batch_size, shuffle=False,
                                           num_workers=num_workers, pin_memory=True, collate_fn=_sparse_collate_fn)
        print('{} ood_illformed examples'.format(len(eval_illformed_data)))

    data_loaders = {
        'train': train_loader,
        'valid': valid_loader,
        'test': test_loader,
        'eval_long': eval_long_loader,
        'eval_new': eval_new_loader,
        'eval_illformed': eval_illformed_loader
    }
    return data_loaders, input_lang, output_lang


def _new_sparse_collate_fn(batch):
    """
    This isn't used at the moment, I tried to get it to work but something weird was going on.
    Once I get this working, text_to_tensors should return lists and not tensors
    """
    input_fillers = []
    input_roles = []
    input_batch = []
    output_fillers = []
    output_roles = []
    output_batch = []
    for i, item in enumerate(batch):
        input_fillers.extend(item['input_fillers'])
        input_roles.extend(item['input_roles'])
        input_batch.extend(len(item['input_fillers']) * [i])
        output_fillers.extend(item['output_fillers'])
        output_roles.extend(item['output_roles'])
        output_batch.extend(len(item['output_fillers']) * [i])
    return {
        'input_fillers': torch.tensor(input_fillers),
        'input_indices': torch.stack((torch.tensor(input_batch), torch.tensor(input_roles))),
        'output_fillers': torch.tensor(output_fillers),
        'output_indices': torch.stack((torch.tensor(output_batch), torch.tensor(output_roles))),
        'batch_size': len(batch)
    }


def _sparse_collate_fn(batch):
    input_fillers = []
    input_roles = []
    output_fillers = []
    output_roles = []
    raw_inputs = []
    raw_outputs = []
    for item in batch:
        input_fillers.append(item['input_fillers'])
        input_roles.append(item['input_roles'])
        output_fillers.append(item['output_fillers'])
        output_roles.append(item['output_roles'])
        raw_inputs.append(item['raw_input'])
        raw_outputs.append(item['raw_output'])
    return {
        'input_fillers': pad_sequence(input_fillers, batch_first=True),
        'input_roles': pad_sequence(input_roles, batch_first=True),
        'output_fillers': pad_sequence(output_fillers, batch_first=True),
        'output_roles': pad_sequence(output_roles, batch_first=True),
        'raw_input': raw_inputs,
        'raw_output': raw_outputs
    }


class BinaryT2TDataset(Dataset):
    """
    Trees are represented as vectors of indices of length 2**depth
    """

    def __init__(self, tsv_file, input_lang, output_lang, max_examples=None, filter_=None, max_depth=20,
                 add_eob_tokens=True, output_lowercase=False, add_eob_to_memory=False, num_extra_tokens_in_memory=0):
        self.max_depth = max_depth
        self.pad_idx = 0
        self.max_input_length = 0
        self.output_lowercase = output_lowercase
        self.add_eob_to_memory = add_eob_to_memory
        self.num_extra_tokens_in_memory = num_extra_tokens_in_memory
        self.input_lang = input_lang
        self.output_lang = output_lang

        with open(tsv_file) as f:
            rows = list(f)

            if filter_ is not None:
                def filter_match(row, dfilter):
                    field3 = row.split("\t")[2].strip()
                    return dfilter.search(field3)

                rows = [row for row in rows if filter_match(row, filter_)]

            if max_examples:
                random.shuffle(rows)
                rows = rows[0:max_examples]

            # print("data rows loaded: {:}".format(len(rows)))
            self.one_input_has_multiple_trees = False
            self.data = self.process_trees(
                rows,
                input_lang,
                output_lang,
                add_eob_tokens=add_eob_tokens,
                is_json=tsv_file.endswith('.json')
            )

    def process_trees(self, data, input_lang, output_lang, add_eob_tokens=True, is_json=False):
        processed = []
        max_branch = 0

        dataset_max_depth = 0

        for line in data:
            if is_json:
                inout_pair = json.loads(line)
                in_nodes = text_tree_to_node(
                    inout_pair['source'],
                    add_eob_tokens=add_eob_tokens,
                    add_eob_to_memory=self.add_eob_to_memory,
                    num_extra_tokens_in_memory=self.num_extra_tokens_in_memory
                )
                if len(in_nodes) > 1:
                    self.one_input_has_multiple_trees = True
                out_str = inout_pair['target']
                if self.output_lowercase:
                    out_str = out_str.replace('I_JUMP', 'jump')
                    out_str = out_str.replace('I_WALK', 'walk')
                    out_str = out_str.replace('I_LOOK', 'look')
                    out_str = out_str.replace('I_RUN', 'run')
                    out_str = out_str.replace('I_TURN_LEFT', 'left')
                    out_str = out_str.replace('I_TURN_RIGHT', 'right')
                out_node = text_tree_to_node(out_str, add_eob_tokens=add_eob_tokens)[0]

                example = {"input": in_nodes, "output": out_node, "example_type": None}
            else:
                inout_pair = line.split('\t')
                tt = None

                if len(inout_pair) > 2:
                    # remove 3rd field used for filtering
                    tt = inout_pair[2].strip()
                    inout_pair = inout_pair[0:2]

                in_nodes = text_tree_to_node(
                    inout_pair[0],
                    add_eob_tokens=add_eob_tokens,
                    add_eob_to_memory=self.add_eob_to_memory,
                    num_extra_tokens_in_memory=self.num_extra_tokens_in_memory,
                )
                if len(in_nodes) > 1:
                    self.one_input_has_multiple_trees = True
                out_node = text_tree_to_node(inout_pair[1], add_eob_tokens=add_eob_tokens)[0]

                example = {"input": in_nodes, "output": out_node, "example_type": tt}

            input_max_depth = 0
            for in_node in in_nodes:
                max_branch = max([max_branch, in_node.get_max_branching()])
                input_max_depth = max([input_max_depth, in_node.get_max_depth()])

            max_branch = max([max_branch, out_node.get_max_branching()])
            assert max_branch <= 2

            if input_max_depth > dataset_max_depth:
                dataset_max_depth = input_max_depth
            if example['output'].get_max_depth() > dataset_max_depth:
                dataset_max_depth = example['output'].get_max_depth()

            if input_max_depth > self.max_depth or example['output'].get_max_depth() > self.max_depth:
                continue

            # add to vocab
            def _add_to_vocab(node, language):
                if node is None:
                    return
                language.add_word(node.label)
                for child in node.children:
                    _add_to_vocab(child, language)
                return

            self.max_input_length = max(self.max_input_length, len(example['input']))
            for input_tree in example['input']:
                _add_to_vocab(input_tree, input_lang)
            _add_to_vocab(example['output'], output_lang)

            processed.append(example)

        print('Max depth seen in file: {}'.format(dataset_max_depth))
        return processed

    def get_direct(self, idx):
        return self.__getitem__(idx)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_roles, input_fillers = self.text_to_tensors(item['input'], self.input_lang)
        output_roles, output_fillers = self.text_to_tensors(item['output'], self.output_lang)
        return {
            'input_fillers': input_fillers,
            'input_roles': input_roles,
            'output_fillers': output_fillers,
            'output_roles': output_roles,
            'raw_input': item['input'],
            'raw_output': item['output']
        }

    def text_to_tensors(self, root, language):
        # We create a sparse tensor by default and cast it to strided when returning if needed
        role_indices = []
        filler_indices = []

        def _traverse_and_tensorify(node, ind, _filler_indices, _role_indices):
            if node is None:
                return
            _role_indices.append(ind)
            _filler_indices.append(language.vocab2ind[node.label])
            if len(node.children) > 0:
                # work on the left child
                _traverse_and_tensorify(node.children[0], left_child(ind), _filler_indices, _role_indices)
            if len(node.children) > 1:
                # work on the right child
                _traverse_and_tensorify(node.children[1], right_child(ind), _filler_indices, _role_indices)
            return

        if type(root) is list:
            for tree in root:
                filler_indices.append([])
                role_indices.append([])
                _traverse_and_tensorify(tree, root_node_index(), filler_indices[-1], role_indices[-1])
        else:
            _traverse_and_tensorify(root, root_node_index(), filler_indices, role_indices)

        role_indices, filler_indices = torch.tensor(role_indices), torch.tensor(filler_indices)
        return (role_indices if self.one_input_has_multiple_trees else role_indices.squeeze(0),
                filler_indices if self.one_input_has_multiple_trees else filler_indices.squeeze(0))

    def __len__(self):
        return len(self.data)


def get_vocab_info(task_path, vocab):
    if 'For2Lam' in task_path:
        binary_vocab = ('<IF>', '<CMP>', '<CMP>|0', '<IF>|0', '<LET>', '<LET>|0', '<LETREC>', '<LETREC>|0',
                        '<LETREC>|1', '<APP>', '<Op+>', '<Op->', '<SEQ>', '<FOR>', '<FOR>|0', '<FOR>|1', '<FOR>|2',
                        '<ASSIGN>')
        unary_vocab = ('<Expr>',)

        # terminal: y, 1, <, 0, x, blank, <UNIT>, func, >, z, ==
        terminal_vocab = list(vocab)
        for word in unary_vocab:
            terminal_vocab.remove(word)
        for word in binary_vocab:
            terminal_vocab.remove(word)
    elif 'car_cdr_rcons' in task_path:
        unary_vocab = ('NOUN', 'DET')
        binary_vocab = ('CAR', 'CDR', 'RCONS', 'R')
        terminal_vocab = list(vocab)
        for word in unary_vocab:
            terminal_vocab.remove(word)
        for word in binary_vocab:
            terminal_vocab.remove(word)
    elif 'active_logical' in task_path:
        unary_vocab = ('N', 'DET', 'ADJ', 'V')
        binary_vocab = ('S', 'NP', 'AP', 'VP', 'LF', 'ARGS')
        terminal_vocab = list(vocab)
        for word in unary_vocab:
            terminal_vocab.remove(word)
        for word in binary_vocab:
            terminal_vocab.remove(word)
    elif 'SCAN' in task_path:
        unary_vocab = ()
        binary_vocab = ('<NT>')
        terminal_vocab = list(vocab)
        terminal_vocab.remove('<NT>')
    elif 'cognition' in task_path:
        unary_vocab = ()
        binary_vocab = ('<NT>')
        terminal_vocab = list(vocab)
        terminal_vocab.remove('<NT>')
    else:
        print(f'{task_path} is not supported in get_vocab_info() yet, returning empty vocab info')
        unary_vocab = ('<PAD>',)
        binary_vocab = ('<PAD>',)
        terminal_vocab = ('<PAD>',)

    vocab_info = {
        'unary': unary_vocab,
        'binary': binary_vocab,
        'terminal': terminal_vocab
    }
    return vocab_info
