import logging
from enum import Enum
from typing import Union, Optional

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from nltk import TreePrettyPrinter, Tree
from torch import Tensor
from torch_geometric.utils import coalesce

from TPR_utils import batch_symbols_to_node_tree, build_D, build_E, SparseTPR, SparseTPRBlock
from sparsemax import Sparsemax
from utils import pashamax, sinusoidal_positional_encoding

sparsemax = Sparsemax(dim=-1)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class RootPredictionType(Enum):
    """
    Enum for how to generate the cons root argument.

    @LINEAR: A linear transformation from d_model->d_filler
    @ATTN_OVER_DICT: A linear transformation from d_model->n_fillers followed by a weighted sum over the dictionary of
      embeddings
    @POSITION_ATTN_OVER_INPUTS: A linear transformation from d_model->input_length followed by a weighted sum over the
      input embeddings
    @QK_ATTN_OVER_INPUTS: A linear transformation from d_model->d_key followed by query-key attention and a weighted
      sum over the input embeddings
    """
    LINEAR = 'LINEAR'
    ATTN_OVER_DICT = 'ATTN_OVER_DICT'
    POSITION_ATTN_OVER_INPUTS = 'POSITION_ATTN_OVER_INPUTS'
    QK_ATTN_OVER_INPUTS = 'QK_ATTN_OVER_INPUTS'

    def __str__(self):
        return self.value


class DiffTreeMachine(nn.Module):
    def __init__(self, config):
        super().__init__()
        d_tpr = config.d_filler * config.d_role

        # The number of bits used to store the role index
        self.role_bits = int(math.log2(config.tpr.num_roles))

        self.filler_map_location = config.filler_map_location
        if config.filler_map_location:
            self.filler_map = nn.Linear(config.d_filler, config.d_filler)
            torch.nn.init.eye_(self.filler_map.weight)
            if config.filler_map_location == 'operation':
                # self.num_ops = 4
                raise NotImplementedError
            else:
                self.num_ops = 3
            if config.filler_map_type != 'linear':
                raise NotImplementedError
        else:
            self.num_ops = 3

        # Controls whether roles are summed over before performing the linear transformation for ctrl_type == 'conv'
        # This is necessary because the size of the linear layer grows with the role dimension which is too large
        # for high depths
        self.sum_over_roles = True
        self.ctrl_type = config.ctrl_type
        if self.ctrl_type == 'linear':
            self.ctrl_net = nn.Linear(d_tpr, config.d_model)
        elif self.ctrl_type == 'conv_mlp':
            self.ctrl_net = nn.Sequential(
                nn.Conv1d(in_channels=config.d_filler, out_channels=config.n_conv_kernels, kernel_size=1, stride=1),
                nn.GELU(),
                nn.Flatten(),
                nn.Linear(config.n_conv_kernels * config.d_role, config.d_model)
            )
        elif self.ctrl_type == 'conv':
            self.ctrl_net = nn.Sequential(
                nn.Conv1d(
                    in_channels=config.d_filler,
                    out_channels=config.n_conv_kernels,
                    kernel_size=1,
                    stride=1,
                    bias=False
                ),
                SumModule(-1) if self.sum_over_roles else nn.Identity(),
                nn.Flatten(),
                nn.Linear(config.n_conv_kernels, config.d_model) if self.sum_over_roles else nn.Linear(
                    config.n_conv_kernels * config.d_role,
                    config.d_model
                    )
            )
        elif self.ctrl_type == 'set_transformer':
            self.ctrl_net = SetTransformer(
                config.d_model,
                config.d_filler,
                self.role_bits,
                config.nhead,
                config.dim_feedforward
            )
        else:
            raise RuntimeError('Unsupported ctrl_type: {}'.format(config.ctrl_type))

        self.encoding_dropout = nn.Dropout(config.dropout)

        self.interpreter = DiffTreeInterpreter(
            tpr=config.tpr,
            num_ops=self.num_ops,
            predefined_operations_are_random=config.predefined_operations_are_random,
            sparse=config.sparse,
            filler_threshold=config.filler_threshold,
            max_filled_roles=config.max_filled_roles,
            cons_only=config.cons_only,
            new_tree_filler_dropout1d=config.new_tree_filler_dropout1d,
            config=config,
        )
        self.max_filled_roles = config.max_filled_roles
        self.max_roles_during_training = config.max_filled_roles
        self.max_roles_during_eval = config.tpr.num_roles

        self.dtm_layers = config.dtm_layers

        config.num_ops = self.num_ops
        config.filler_matrix = config.tpr.out.weight

        self.nta = NeuralTreeAgent(config)
        self.op_logits_token = nn.parameter.Parameter(torch.Tensor(1, config.d_model))
        nn.init.normal_(self.op_logits_token)
        self.root_filler_token = nn.parameter.Parameter(torch.Tensor(1, config.d_model))
        nn.init.normal_(self.root_filler_token)

        # We only need positional embeddings for inputs with multiple trees
        if config.max_input_length > 1:
            if config.random_positional_max_len > config.max_input_length:
                max_len = config.random_positional_max_len
                self.random_positional_embedding = True
            else:
                max_len = config.max_input_length
                self.random_positional_embedding = False
            uniform_weights = torch.zeros(max_len).softmax(0)
            self.register_buffer('uniform_weights', uniform_weights)
            if config.positional_embedding_type == 'learned':
                # TODO: what should be the scale and initialization of the embeddings?
                positional_embeddings = nn.Embedding(max_len, config.d_model)
                self.register_parameter('positional_embeddings', positional_embeddings.weight)
            elif config.positional_embedding_type == 'sinusoidal':
                # TODO: also, check the norm of the positional embeddings. Does the sinusoidal embeddings need to be
                #  scaled?
                positional_embeddings = torch.tensor(
                    sinusoidal_positional_encoding(max_len, config.d_model),
                    dtype=torch.float32
                )
                self.register_buffer('positional_embeddings', positional_embeddings)
            else:
                raise RuntimeError('Unsupported positional_embedding_type: {}'.format(config.positional_embedding_type))
        else:
            self.positional_embeddings = None

        # input_lang and output_lang will be used for debugging in forward()
        self.input_lang = config.input_lang
        self.output_lang = config.output_lang
        self.tpr = config.tpr

        self.include_empty_tpr = config.include_empty_tpr
        self.sparse = config.sparse
        self.cons_only = config.cons_only

        self.filler_dropout_location = config.filler_dropout_location
        if config.filler_dropout_location:
            self.filler_dropout = nn.Dropout(config.filler_dropout)

        self.filler_noise_location = config.filler_noise_location
        self.filler_noise_std = config.filler_noise_std

        self.root_prediction_type = config.root_prediction_type

    def forward(self, input_tpr, bsz, debug=False, calculate_entropy=False, custom_memory_set=False,
                decoded_tpr_to_tree_fn=None, vocab_info=None, input_filler_root_embeddings=None,
                input_filler_root_mask=None
    ) -> (Union[SparseTPR, Tensor], Optional[dict], None):
        debug_writer = [] if debug else None

        if not self.sparse:
            has_multiple_input_tprs = input_tpr.dim() == 4

        if self.filler_dropout_location == 'input':
            input_tpr = SparseTPRBlock(input_tpr.indices(), self.filler_dropout(input_tpr.values()))

        # TODO: Question, dropout rescales the values during training to keep the same magnitude. Should we do
        #  something similar when we add noise?
        if self.filler_noise_location == 'input' and self.training:
            # We don't want to add noise to the padding embeddings
            if self.sparse:
                pad_mask = (input_tpr.values() == 0).all(dim=1)
                noisy_values = input_tpr.values() + torch.randn_like(input_tpr.values()) * self.filler_noise_std
                noisy_values.masked_fill_(pad_mask.unsqueeze(1), 0)
                input_tpr = SparseTPRBlock(input_tpr.indices(), noisy_values)
            else:
                pad_mask = (input_tpr == 0).all(dim=-1)
                noisy_values = input_tpr + torch.randn_like(input_tpr) * self.filler_noise_std
                noisy_values.masked_fill_(pad_mask.unsqueeze(-1), 0)
                input_tpr = noisy_values

        # We need to convert indices to words for printing
        if debug:
            assert self.input_lang
            assert self.output_lang

        if self.filler_map_location == 'pre_dtm':
            if self.sparse:
                input_fillers = self.apply_sparse_filler_transformation(input_tpr.values())
                input_tpr = torch.sparse_coo_tensor(indices=input_tpr.indices(), values=input_fillers,
                                                    size=input_tpr.shape).coalesce()
            else:
                input_tpr = self.apply_dense_filler_transformation(input_tpr)

        if self.sparse:
            step_offset = input_tpr.memory_slot_indices().max() + 1 if self.include_empty_tpr \
                else input_tpr.memory_slot_indices().max()
        else:
            step_offset = input_tpr.shape[1] - 1
        if custom_memory_set:
            # Add 1 for the input TPR
            memory_slots = self.dtm_layers + 1 + step_offset
            if self.sparse:
                values_storage = torch.zeros((bsz * memory_slots * self.max_filled_roles, self.tpr.d_filler),
                                              device=input_tpr.device)
                # The 3 indices are batch, memory slot, role
                indices_storage = torch.zeros((3, bsz * memory_slots * self.max_filled_roles),
                                               device=input_tpr.device, dtype=torch.int32)
                # The start and end indices for each storage slot. For example, if the first layer writes to memory
                # index 0 in dimensions 0-71, the second memory slot would write from 71 onwards. This tensor keeps
                # track of where each memory slot starts and ends in order to know where to write the next TPR as
                # well as how to retrieve the TPRs from storage.

                # We add 2 to the number of dtm_layers because we need to include a starting index (index 0) and the input
                # TPRs which end at index 1.
                storage_start_end_indices = torch.zeros(self.dtm_layers + 2, device=input_tpr.device, dtype=torch.long)

                indices_storage, values_storage = sparse_storage_block_set(
                    indices_storage,
                    values_storage,
                    input_tpr.indices(),
                    input_tpr.values(),
                    0,
                    storage_start_end_indices
                )
                if self.include_empty_tpr:
                    raise NotImplementedError('include_empty_tpr does not work with sparse and custom_memory_set yet')
            else:
                # Include one extra for the input TPR and a potential extra for the 0 TPR
                memory = torch.zeros((bsz, memory_slots, input_tpr.shape[-2], input_tpr.shape[-1]),
                                     device=input_tpr.device, layout=torch.sparse_coo if self.sparse else torch.strided)
                memory = memory_set(memory, input_tpr, 0)
                if self.include_empty_tpr:
                    memory = memory_set(memory, self.tpr.empty_tpr(input_tpr.device), 1)
        else:
            if self.sparse:
                memory = torch.sparse_coo_tensor(indices=torch.stack((
                    input_tpr.indices()[0],
                    torch.zeros(input_tpr.values().shape[0], device=input_tpr.device),
                    input_tpr.indices()[1]
                )), values=input_tpr.values(), size=(
                    bsz, 2 if self.include_empty_tpr else 1, self.tpr.num_roles, self.tpr.d_filler)
                ).coalesce()
            else:
                memory = input_tpr.unsqueeze(1)  # the dtm_layers dimension

        # TODO: if self.cons_only==True, we don't need the operation token
        # Setup the encodings for the NTA
        op_logits_token = self.op_logits_token.repeat(bsz, 1, 1)
        root_filler_token = self.root_filler_token.repeat(bsz, 1, 1)

        encodings = torch.cat((op_logits_token, root_filler_token), dim=1)

        # Encode the trees that initialize the memory
        if self.sparse:
            # The input trees always have root nodes, so we can count the number of root nodes to get the number of trees
            trees_encoding, agent_pad_mask = self.ctrl_net(input_tpr, bsz, num_trees=(input_tpr.role_indices() == 1).sum())
        else:
            if has_multiple_input_tprs:
                trees_encoding = self.ctrl_net(input_tpr.view(input_tpr.shape[0], input_tpr.shape[1], -1))
                agent_pad_mask = (input_tpr == 0).all(dim=-1).all(dim=-1)
            else:
                trees_encoding = self.ctrl_net(input_tpr.view(input_tpr.shape[0], -1))
                agent_pad_mask = (input_tpr == 0).all(dim=-1)
        # Add two non-padding tokens to the agent pad mask for the op logits and root filler tokens
        agent_pad_mask = torch.cat((torch.zeros((bsz, 2), device=agent_pad_mask.device, dtype=torch.bool),
                                   agent_pad_mask), dim=1)

        # TODO: technically, I don't think we need to add positional_embeddings to the <EOB> token
        if self.positional_embeddings is not None:
            if self.random_positional_embedding:# and self.training:
                positional_indices = torch.multinomial(self.uniform_weights, trees_encoding.shape[1], replacement=False)
                positional_indices = positional_indices.sort()[0]
            else:
                positional_indices = torch.arange(trees_encoding.shape[1], device=trees_encoding.device)
            positional_embeddings = torch.index_select(self.positional_embeddings, 0, positional_indices).unsqueeze(0)
        else:
            positional_embeddings = torch.tensor([0], device=trees_encoding.device)

        encodings = torch.cat(
            (
                encodings,
                self.encoding_dropout(trees_encoding + positional_embeddings)
            ),
            dim=1
        )

        # Step offset will cause us to skip shrinking the input TPR in the first loop, so we encode that here if we
        # include the empty TPR
        if self.include_empty_tpr:
            raise RuntimeError('I need to rethink how empty TPRs work, maybe replace it with a root node with the EOB'
                               'filler?')
            if self.filler_map_location == 'pre_shrink':
                # Remove the memory slot dimension
                if self.sparse:
                    transformed_fillers = self.apply_sparse_filler_transformation(input_tpr.values())
                    tree_to_shrink = torch.sparse_coo_tensor(indices=memory.indices()[[0, 2]],
                                                             values=transformed_fillers,
                                                             size=(memory.shape[0], memory.shape[2],
                                                                   memory.shape[3])).coalesce()
                else:
                    tree_to_shrink = self.apply_dense_filler_transformation(memory[:, 0])
            else:
                if self.sparse:
                    tree_to_shrink = input_tpr
                else:
                    tree_to_shrink = memory.select(1, 0)

            if self.ctrl_type == 'linear':
                tree_encoding = self.ctrl_net(tree_to_shrink.flatten(1))
            elif self.ctrl_type == 'conv_mlp' or self.ctrl_type == 'conv' or self.ctrl_type == 'set_transformer':
                tree_encoding = self.ctrl_net(tree_to_shrink)
            else:
                raise RuntimeError('Unsupported ctrl_type: {}'.format(self.ctrl_type))
            encodings = torch.cat((encodings, self.encoding_dropout(tree_encoding).unsqueeze(1)), dim=1)

        cons_arg1_entropies = torch.empty(self.dtm_layers, device=input_tpr.device)
        cons_arg2_entropies = torch.empty(self.dtm_layers, device=input_tpr.device)
        for step in range(self.dtm_layers):
            # logger.verbose(f'Layer {step} allocated memory {torch.cuda.memory_allocated() / 1024 ** 2} mb')

            # Encode the most recent TPR in memory except at step 0 where we encoded the inputs above
            if step > 0:
                if self.filler_map_location == 'pre_shrink':
                    if self.sparse:
                        most_recent_mask = memory.indices()[1] == step + step_offset
                        transformed_fillers = self.apply_sparse_filler_transformation(memory.values()[most_recent_mask])
                        tree_to_shrink = torch.sparse_coo_tensor(indices=memory.indices()[[0, 2]][:, most_recent_mask],
                                                                 values=transformed_fillers,
                                                                 size=(memory.shape[0], memory.shape[2],
                                                                       memory.shape[3])).coalesce()
                    else:
                        tree_to_shrink = self.apply_dense_filler_transformation(memory[:, step + step_offset])
                else:
                    if self.sparse:
                        if custom_memory_set:
                            tree_to_shrink = get_sparse_tpr_from_storage(
                                indices_storage,
                                values_storage,
                                step,
                                storage_start_end_indices
                            )
                        else:
                            most_recent_memory_mask = memory.indices()[1] == step + step_offset
                            tree_to_shrink = torch.sparse_coo_tensor(indices=torch.stack(
                                (memory.indices()[0][most_recent_memory_mask], memory.indices()[-1][most_recent_memory_mask])),
                                values=memory.values()[most_recent_memory_mask], size=(bsz, self.tpr.num_roles,
                                                                                       self.tpr.d_filler)).coalesce()
                    else:
                        tree_to_shrink = memory.select(1, step + step_offset)
                if self.ctrl_type == 'linear':
                    tree_encoding = self.ctrl_net(tree_to_shrink.flatten(1))
                    pad = torch.zeros((bsz, 1), dtype=torch.bool, device=tree_encoding.device)
                elif self.ctrl_type == 'conv_mlp' or self.ctrl_type == 'conv':
                    if self.sparse:
                        # The output of the conv layer is batch_size, n_kernels, n_roles. As n_roles gets very large,
                        # this can be very memory intensive. We can avoid this by summing over the role dimension first
                        # so that the output is batch_size, n_kernels. TODO: look into using SetTransformer with a learned
                        # query vector to perform weighted summing over the roles
                        if self.sum_over_roles:
                            summed_over_roles = torch.sparse.sum(tree_to_shrink, dim=1).to_dense()
                            num_filled_roles = torch.bincount(tree_to_shrink.indices()[0])
                            # Normalize the result by dividing by the number of filled roles
                            normalized = torch.div(summed_over_roles, num_filled_roles[None].T)
                            conv_results = normalized @ self.ctrl_net[0].weight.transpose(0, 1).squeeze()
                            tree_encoding = self.ctrl_net[3](conv_results)
                        else:
                            conv_results = tree_to_shrink.values() @ self.ctrl_net[0].weight.transpose(0, 1).squeeze()
                            # sparse_indices[0] is the batch index, sparse_indices[1] is the role index
                            sparse_indices = tree_to_shrink.indices()
                            # batch_size, n_kernels, n_roles
                            dense_conv_results = torch.zeros((bsz, self.ctrl_net[0].out_channels, input_tpr.shape[1]),
                                                             device=input_tpr.device)
                            dense_conv_results[sparse_indices[0], ..., sparse_indices[1]] = conv_results
                            flat = self.ctrl_net[2](dense_conv_results)
                            tree_encoding = self.ctrl_net[3](flat)
                    else:
                        tree_encoding = self.ctrl_net(tree_to_shrink)
                elif self.ctrl_type == 'set_transformer':
                    if self.sparse:
                        tree_encoding, pad = self.ctrl_net.single_tree_forward(tree_to_shrink, bsz)
                    else:
                        raise RuntimeError('Set Transformer shrink only implemented for sparse TPRs.')
                else:
                    raise RuntimeError('Unsupported ctrl_type: {}'.format(self.ctrl_type))
                encodings = torch.cat(
                    (encodings,
                     self.encoding_dropout(tree_encoding.view(bsz, 1, -1))), dim=1)
                agent_pad_mask = torch.cat((agent_pad_mask, pad), dim=1)

            op_dist, root_filler, arg_weights, encodings, root_filler_dist = self.nta(
                encodings,
                step,
                agent_pad_mask,
                input_filler_root_embeddings=input_filler_root_embeddings,
                input_filler_root_mask=input_filler_root_mask
            )
            # TODO: now that the arg weights are produced by the NTA, we can use them to calculate the entropy of the
            # distributions here instead of in the separate car/cdr/cons nets

            if self.cons_only:
                op_dist = torch.zeros_like(op_dist)
                op_dist[:, 2] = 1

            cons_arg1_entropies[step] = -torch.sum(
                arg_weights[:, :, 2] * torch.log(arg_weights[:, :, 2] + 1e-9),
                dim=1
            ).mean()  # Adding a small constant for numerical stability
            cons_arg2_entropies[step] = -torch.sum(
                arg_weights[:, :, 3] * torch.log(arg_weights[:, :, 3] + 1e-9),
                dim=1
            ).mean()  # Adding a small constant for numerical stability

            use_checkpoint = False
            if use_checkpoint:
                new_tree = checkpoint.checkpoint(self.interpreter, memory[:, :step + 1 + step_offset], arg_weights,
                                                 root_filler, op_dist)
            else:
                if custom_memory_set:
                    if self.sparse:
                        trees_in_memory = get_sparse_tpr_block_from_storage(
                            indices_storage,
                            values_storage,
                            storage_end_index=step+1,
                            storage_start_end_indices=storage_start_end_indices
                        )
                        new_tree = self.interpreter(
                            trees_in_memory,
                            arg_weights,
                            root_filler,
                            op_dist,
                            bsz,
                            # Don't perform dropout on the last step before the output
                            skip_dropout=step == self.dtm_layers - 1
                        )
                    else:
                        new_tree = self.interpreter(memory[:, :step + 1 + step_offset], arg_weights, root_filler,
                            op_dist, bsz)
                else:
                    new_tree = self.interpreter(memory, arg_weights, root_filler, op_dist, bsz)

            if debug:
                output_string = 'Step {}:\nBlackboard:'.format(step)
                debug_writer.append(output_string)
                # Use the batch dimension to decode previous layers on the blackboard
                if custom_memory_set and self.sparse:
                    trees_to_decode = trees_in_memory
                    first_batch_mask = trees_to_decode.batch_indices() == 0
                    # Ignore the batch index dimension by indexing from [1:]
                    trees_to_decode = SparseTPR(trees_to_decode.indices()[1:][:, first_batch_mask],
                                                trees_to_decode.values()[first_batch_mask])
                else:
                    trees_to_decode = memory[0, :step + 1 + step_offset]
                if self.sparse and not custom_memory_set:
                    trees_to_decode = trees_to_decode.coalesce()

                if self.sparse:
                    x_decoded = decoded_tpr_to_tree_fn(
                        self.tpr.unbind(
                            (SparseTPR(trees_to_decode.indices(), trees_to_decode.values())), decode=True, type_='input'
                        )
                    )
                else:
                    x_decoded = decoded_tpr_to_tree_fn(
                        self.tpr.unbind(trees_to_decode, decode=True, type_='input')
                    )
                x_tree = batch_symbols_to_node_tree(
                    x_decoded,
                    self.input_lang.ind2vocab,
                    terminal_vocab=vocab_info['terminal'],
                    unary_vocab=vocab_info['unary'],
                    sparse=self.sparse
                )
                extra_column_label = ''
                is_root_prediction_attn = ((self.root_prediction_type == RootPredictionType.ATTN_OVER_DICT or
                        self.root_prediction_type == RootPredictionType.QK_ATTN_OVER_INPUTS) and
                        not self.nta.hardcode_cons_root_index)
                if is_root_prediction_attn:
                    extra_column_label = ', root filler'
                if self.cons_only:
                    debug_writer.append(f'[cons_l, cons_r{extra_column_label}]')
                else:
                    debug_writer.append(f'[car, cdr, cons_l, cons_r{extra_column_label}]')
                for i, tree in enumerate(x_tree):
                    if tree:
                        if self.cons_only:
                            weights_string = np.array2string(
                                arg_weights[0, i, -2:].detach().cpu().numpy(),
                                formatter={'float_kind': arg_weight_formatter}
                            )
                        else:
                            weights_string = np.array2string(
                                arg_weights[0, i, :].detach().cpu().numpy(),
                                formatter={'float_kind': arg_weight_formatter}
                            )
                        # Only print the input root attention for valid input fillers in the 0th batch
                        if (is_root_prediction_attn and input_filler_root_mask and
                                i < input_filler_root_mask[0].shape[0] and not input_filler_root_mask[0, i]):
                            root_attn = f'{root_filler_dist[0, i]:.2f}'[1:]
                            weights_string = weights_string[:-1] + f' {root_attn}' + ']'
                        if i > step_offset:
                            debug_writer.append(f'{i:2}\t{weights_string} {i-step_offset-1:2}. {tree.str()}')
                        else:
                            debug_writer.append(f'{i:2}\t{weights_string}    {tree.str()}')
                    else:
                        debug_writer.append('None')

                if not self.cons_only:
                    debug_writer.append(
                        'car: {:.3f}\tcdr: {:.3f}\tcons: {:.3f}'.format(op_dist[0][0], op_dist[0][1], op_dist[0][2]))
                if (not self.nta.hardcode_cons_root_index and self.root_prediction_type ==
                        RootPredictionType.ATTN_OVER_DICT):
                    root_filler_dist_str = 'root filler ~ '
                    for i, filler_weight in enumerate(root_filler_dist[0]):
                        if filler_weight > .1:
                            root_filler_dist_str += f'{self.output_lang.ind2vocab[i]}: {filler_weight:.2}\t'
                    debug_writer.append(root_filler_dist_str)
                tree_to_decode = new_tree[0].unsqueeze(0)
                if self.sparse:
                    tree_to_decode = tree_to_decode.coalesce()
                fully_decoded = decoded_tpr_to_tree_fn(
                    self.tpr.unbind(
                        SparseTPR(tree_to_decode.indices(), tree_to_decode.values()) if self.sparse else tree_to_decode,
                        decode=True,
                        type_='output' if step == self.dtm_layers - 1 else 'input'
                    )
                )
                debug_tree = batch_symbols_to_node_tree(
                    fully_decoded,
                    self.output_lang.ind2vocab if step == self.dtm_layers - 1 else self.input_lang.ind2vocab,
                    terminal_vocab=vocab_info['terminal'],
                    unary_vocab=vocab_info['unary'],
                    sparse=self.sparse
                )[0]
                debug_writer.append('Output: ')
                if not debug_tree:
                    debug_writer.append('None')
                else:
                    pretty_tree = TreePrettyPrinter(Tree.fromstring(debug_tree.str()))
                    debug_writer.append('```{}```'.format(pretty_tree.text()))

            if custom_memory_set:
                if self.sparse:
                    indices_storage, values_storage = sparse_storage_set(
                        indices_storage,
                        values_storage,
                        new_tree.indices(),
                        new_tree.values(),
                        step + 1,
                        step + 1 + step_offset,
                        storage_start_end_indices
                    )
                else:
                    memory = memory_set(memory, new_tree, step + 1 + step_offset)
            else:
                # LARGE MEMORY USAGE
                if self.sparse:
                    memory = torch.sparse_coo_tensor(indices=torch.stack((
                        torch.cat((memory.indices()[0], new_tree.indices()[0])),
                        torch.cat((memory.indices()[1],
                                   torch.tensor(step + step_offset + 1, device=input_tpr.device).repeat(
                                       new_tree._nnz()))),
                        torch.cat((memory.indices()[2], new_tree.indices()[1])))),
                        values=torch.cat((memory.values(), new_tree.values())),
                        size=(memory.shape[0], memory.shape[1] + 1, memory.shape[2], memory.shape[3])).coalesce()
                else:
                    memory = torch.cat([memory, new_tree.unsqueeze(1)], dim=1)

        if self.filler_map_location == 'post_dtm' or self.filler_map_location == 'pre_shrink':
            if self.sparse:
                if custom_memory_set:
                    output = get_sparse_tpr_from_storage(
                        indices_storage,
                        values_storage,
                        step + 1,
                        storage_start_end_indices
                    )
                    output = SparseTPR(output.indices(), self.apply_sparse_filler_transformation(output.values()))
                else:
                    last_memory_mask = memory.indices()[1] == self.dtm_layers + step_offset
                    transformed_fillers = self.apply_sparse_filler_transformation(memory.values()[last_memory_mask])
                    output = torch.sparse_coo_tensor(indices=memory.indices()[[0, 2]][:, last_memory_mask],
                                                     values=transformed_fillers,
                                                     size=input_tpr.shape).coalesce()
            else:
                output = self.apply_dense_filler_transformation(memory[:, self.dtm_layers + step_offset])

            if debug:
                if self.sparse and custom_memory_set:
                    to_decode = output
                    first_batch_mask = output.batch_indices() == 0
                    to_decode = SparseTPR(
                        to_decode.indices()[:, first_batch_mask],
                        to_decode.values()[first_batch_mask]
                    )
                else:
                    to_decode = output[0].unsqueeze(0)

                    if self.sparse:
                        to_decode = to_decode.coalesce()
                fully_decoded = decoded_tpr_to_tree_fn(self.tpr.unbind(to_decode, decode=True, type_='output'))
                debug_tree = batch_symbols_to_node_tree(
                    fully_decoded,
                    self.output_lang.ind2vocab,
                    terminal_vocab=vocab_info['terminal'],
                    unary_vocab=vocab_info['unary']
                )[0]
                debug_writer.append('Post-Linear Output: ')
                if not debug_tree:
                    debug_writer.append('None')
                else:
                    pretty_tree = TreePrettyPrinter(Tree.fromstring(debug_tree.str()))
                    debug_writer.append('```{}```'.format(pretty_tree.text()))
        else:
            if self.sparse:
                if custom_memory_set:
                    output = get_sparse_tpr_from_storage(indices_storage, values_storage, step + 1,
                                                         storage_start_end_indices)
                else:
                    last_memory_mask = memory.indices()[1] == memory.shape[1] - 1
                    output = torch.sparse_coo_tensor(indices=torch.stack(
                        (memory.indices()[0][last_memory_mask], memory.indices()[-1][last_memory_mask])),
                                                     values=memory.values()[last_memory_mask],
                                                     size=(bsz, self.tpr.num_roles, self.tpr.d_filler)).coalesce()
            else:
                # Select the final TPR in memory
                output = memory.select(1, -1)

        debug_info = None
        if debug:
            print('\n'.join(debug_writer))
            debug_info = {'text': debug_writer}

        entropies = {
            'cons_arg1': cons_arg1_entropies,
            'cons_arg2': cons_arg2_entropies
        }

        if self.filler_dropout_location == 'pre_output':
            output = SparseTPRBlock(output.indices(), self.filler_dropout(output.values()))

        return SparseTPR(output.indices(), output.values()) if self.sparse else output, debug_info, entropies

    def set_gumbel_temp(self, temp):
        self.interpreter.gumbel_temp = temp
        self.nta.gumbel_temp = temp

    def apply_sparse_filler_transformation(self, fillers, apply_bias=False):
        """
        Apply a linear transformation on the fillers. The bias is turned off by default since the bias will "turn on"
        empty fillers by making them not zero.
        """
        filler_transformed = fillers @ self.filler_map.weight
        if apply_bias:
            filler_transformed += self.filler_map.bias
        return filler_transformed

    def apply_dense_filler_transformation(self, tpr, apply_bias=False):
        """
        Apply a linear transformation on the fillers. The bias is turned off by default since the bias will "turn on"
        empty fillers by making them not zero.
        """
        filler_transformed = torch.einsum('bfr,ft->btr', tpr, self.filler_map.weight)
        if apply_bias:
            filler_transformed += self.filler_map.bias.unsqueeze(0).unsqueeze(-1)
        return filler_transformed


class NeuralTreeAgent(nn.Module):
    """
    The Neural Tree Agent
    """
    def __init__(self, config):
        super().__init__()
        # We only need to create a single layer since this layer will be deep copied by nn.TransformerEncoder
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation=config.activation,
            layer_norm_eps=config.layer_norm_eps,
            batch_first=True,
            norm_first=bool(config.transformer_norm_first)
        )

        self.is_agent_universal = config.is_agent_universal

        self.filler_matrix = config.filler_matrix
        self.root_prediction_type = config.root_prediction_type
        self.hardcode_cons_root_index = config.hardcode_cons_root_index

        self.layers = nn.ModuleList()
        self.arg_logits_list = nn.ModuleList()
        self.root_filler_list = nn.ModuleList()
        self.op_logits_list = nn.ModuleList()
        self.root_prediction_key_list = nn.ModuleList()
        self.root_prediction_query_list = nn.ModuleList()
        if self.is_agent_universal:
            encoder_norm = nn.LayerNorm(
                config.d_model,
                eps=config.layer_norm_eps
            ) if config.transformer_norm_first else None
            self.layers.append(
                nn.TransformerEncoder(transformer_layer, config.agent_layers_per_step, encoder_norm)
            )
            # 4 for the 4 arguments, car, cdr, cons1, cons2
            arg_logits = nn.Linear(config.d_model, 4)
            nn.init.normal_(arg_logits.weight, std=0.02)
            nn.init.zeros_(arg_logits.bias)
            self.arg_logits_list.append(arg_logits)

            if self.root_prediction_type == RootPredictionType.ATTN_OVER_DICT:
                root_filler = nn.Linear(config.d_model, config.filler_matrix.shape[0])
                nn.init.normal_(root_filler.weight, std=0.02)
                nn.init.zeros_(root_filler.bias)
            elif self.root_prediction_type == RootPredictionType.POSITION_ATTN_OVER_INPUTS:
                # TODO: How should this linear layer be initialized?
                root_filler = nn.Linear(config.d_model, config.max_input_length)
                self.max_input_length = config.max_input_length
            elif self.root_prediction_type == RootPredictionType.QK_ATTN_OVER_INPUTS:
                # root_prediction_key converts fillers to keys
                self.root_prediction_key = nn.Linear(config.d_filler, config.d_model // config.nhead)
                # root_prediction_query converts the root token to a query
                self.root_prediction_query = nn.Linear(config.d_model, config.d_model // config.nhead)
                # Note, the values are the fillers themselves
                root_filler = None
                self.root_prediction_key_list.append(self.root_prediction_key)
                self.root_prediction_query_list.append(self.root_prediction_query)
            elif self.root_prediction_type == RootPredictionType.LINEAR:
                root_filler = nn.Linear(config.d_model, config.d_filler)
            else:
                raise RuntimeError('Unsupported root_prediction_type: {}'.format(self.root_prediction_type))
            self.root_filler_list.append(root_filler)

            op_logits = nn.Linear(config.d_model, config.num_ops)
            nn.init.normal_(op_logits.weight, std=0.02)
            nn.init.zeros_(op_logits.bias)

            self.op_logits_list.append(op_logits)
        else:
            for i in range(config.dtm_layers):
                encoder_norm = nn.LayerNorm(
                    config.d_model,
                    eps=config.layer_norm_eps
                ) if config.transformer_norm_first else None
                self.layers.append(
                    nn.TransformerEncoder(transformer_layer, config.agent_layers_per_step, encoder_norm)
                )

                # 4 for the 4 arguments, car, cdr, cons1, cons2
                arg_logits = nn.Linear(config.d_model, 4)
                nn.init.normal_(arg_logits.weight, std=0.02)
                nn.init.zeros_(arg_logits.bias)
                self.arg_logits_list.append(arg_logits)

                if self.root_prediction_type == RootPredictionType.ATTN_OVER_DICT:
                    root_filler = nn.Linear(config.d_model, config.filler_matrix.shape[0])
                    nn.init.normal_(root_filler.weight, std=0.02)
                    nn.init.zeros_(root_filler.bias)
                elif self.root_prediction_type == RootPredictionType.POSITION_ATTN_OVER_INPUTS:
                    # TODO: How should this linear layer be initialized?
                    root_filler = nn.Linear(config.d_model, config.max_input_length)
                    self.max_input_length = config.max_input_length
                elif self.root_prediction_type == RootPredictionType.QK_ATTN_OVER_INPUTS:
                    # root_prediction_key converts fillers to keys
                    self.root_prediction_key = nn.Linear(config.d_filler, config.d_model // config.nhead)
                    # root_prediction_query converts the root token to a query
                    self.root_prediction_query = nn.Linear(config.d_model, config.d_model // config.nhead)
                    # Note, the values are the fillers themselves
                    root_filler = None
                    self.root_prediction_key_list.append(self.root_prediction_key)
                    self.root_prediction_query_list.append(self.root_prediction_query)
                elif self.root_prediction_type == RootPredictionType.LINEAR:
                    root_filler = nn.Linear(config.d_model, config.d_filler)
                else:
                    raise RuntimeError('Unsupported root_prediction_type: {}'.format(self.root_prediction_type))
                self.root_filler_list.append(root_filler)

                op_logits = nn.Linear(config.d_model, config.num_ops)
                nn.init.normal_(op_logits.weight, std=0.02)
                nn.init.zeros_(op_logits.bias)

                self.op_logits_list.append(op_logits)

        self.op_dist_fn = config.op_dist_fn
        self.arg_dist_fn = config.arg_dist_fn
        self.pad_idx = config.pad_idx
        self.op_token_idx = 0
        self.root_filler_token_idx = 1
        self.arg_noise_std = config.arg_noise_std

    def forward(
        self,
        encodings,
        step,
        pad_mask=None,
        input_filler_root_embeddings=None,
        input_filler_root_mask=None
    ):
        if self.is_agent_universal:
            step = 0
        encodings = self.layers[step](encodings, src_key_padding_mask=pad_mask)

        op_logits = self.op_logits_list[step](encodings[:, self.op_token_idx, :])
        if self.op_dist_fn == 'softmax':
            op_dist = F.softmax(op_logits, dim=-1)
        elif self.op_dist_fn == 'pashamax':
            op_dist = pashamax(op_logits, dim=-1)
        elif self.op_dist_fn == 'sparsemax':
            op_dist = sparsemax(op_logits)
        elif self.op_dist_fn == 'gumbel':
            op_dist = F.gumbel_softmax(op_logits, tau=self.gumbel_temp, hard=True)
        else:
            raise ValueError('Unknown op_dist_fn: {}'.format(self.op_dist_fn))

        root_filler_dist = None
        if self.hardcode_cons_root_index:
            if self.hardcode_cons_root_index == -1:
                root_filler = torch.zeros((encodings.shape[0], self.filler_matrix.shape[-1]), device=encodings.device)
            else:
                root_filler = self.filler_matrix[self.hardcode_cons_root_index].expand(encodings.shape[0], -1)
        else:
            if self.root_prediction_type == RootPredictionType.ATTN_OVER_DICT:
                root_filler_logits = self.root_filler_list[step](encodings[:, self.root_filler_token_idx, :])
                # We don't ever want to predict the padding token. Also, this prevents gradient from following through the
                # padding embedding which would make it non-zero.
                root_filler_logits[:, self.pad_idx] = -float('inf')

                root_filler_dist = F.softmax(root_filler_logits, dim=-1)
                # batch x n_fillers, n_fillers x d_filler
                root_filler = torch.einsum('bn,nd->bd', root_filler_dist, self.filler_matrix)
            elif self.root_prediction_type == RootPredictionType.POSITION_ATTN_OVER_INPUTS:
                root_filler_logits = self.root_filler_list[step](encodings[:, self.root_filler_token_idx, :])

                # Mask padding tokens -inf
                root_filler_logits.masked_fill_(input_filler_root_mask, -float('inf'))
                # Softmax
                root_filler_dist = root_filler_logits.softmax(dim=-1)
                # batch x max_input_length, batch x max_input_length x d_filler
                root_filler = torch.einsum('bm,bmd->bd', root_filler_dist, input_filler_root_embeddings)
            elif self.root_prediction_type == RootPredictionType.QK_ATTN_OVER_INPUTS:
                # TODO: for a universal agent, the keys can be cached instead of recomputed
                keys = self.root_prediction_key_list[step](input_filler_root_embeddings)
                query = self.root_prediction_query_list[step](encodings[:, self.root_filler_token_idx, :])

                # batch x n_input_fillers x d_key, batch d_query
                query_key_match = torch.einsum('blk,bk->bl', keys, query)
                query_key_match.masked_fill_(input_filler_root_mask, -float('inf'))
                root_filler_dist = F.softmax(query_key_match / np.sqrt(keys.shape[-1]), dim=1)

                root_filler = torch.einsum('blv,bl->bv', input_filler_root_embeddings, root_filler_dist)
            elif self.root_prediction_type == RootPredictionType.LINEAR:
                root_filler = self.root_filler_list[step](encodings[:, self.root_filler_token_idx, :])
            else:
                raise RuntimeError('Unsupported root_prediction_type: {}'.format(self.root_prediction_type))

        arg_logits = self.arg_logits_list[step](encodings[:, 2:, :])

        if self.arg_noise_std != 0 and self.training:
            arg_logits = arg_logits + torch.randn_like(arg_logits) * self.arg_noise_std

        # Pad mask tracks which tokens are padding, so in this case we need to flip the boolean value to keep the
        # correct values
        arg_logits = torch.where(~pad_mask[:, 2:].unsqueeze(-1), arg_logits, -1e9)
        if self.arg_dist_fn == 'softmax':
            arg_weights = F.softmax(arg_logits, dim=1)
        elif self.arg_dist_fn == 'gumbel':
            arg_weights = F.gumbel_softmax(arg_logits, tau=self.gumbel_temp)
        else:
            raise ValueError('Unknown arg_dist_fn: {}'.format(self.arg_dist_fn))

        quantize = False
        if quantize:
            max_values = arg_weights.max(dim=1, keepdim=True).values
            arg_weights = (arg_weights == max_values).float()

        return op_dist, root_filler, arg_weights, encodings, root_filler_dist


class DiffTreeInterpreter(nn.Module):
    def __init__(self, tpr, num_ops=3, predefined_operations_are_random=False, sparse=False, filler_threshold=None,
                 max_filled_roles=None, cons_only=False, new_tree_filler_dropout1d=0., config=None):
        super().__init__()

        role_matrix = tpr.role_matrix
        if predefined_operations_are_random:
            d_role = role_matrix.shape[1]
            D_l = nn.Parameter(role_matrix.new_empty(d_role, d_role))
            D_r = nn.Parameter(role_matrix.new_empty(d_role, d_role))
            E_l = nn.Parameter(role_matrix.new_empty(d_role, d_role))
            E_r = nn.Parameter(role_matrix.new_empty(d_role, d_role))
            nn.init.kaiming_uniform_(D_l, a=math.sqrt(5))
            nn.init.kaiming_uniform_(D_r, a=math.sqrt(5))
            nn.init.kaiming_uniform_(E_l, a=math.sqrt(5))
            nn.init.kaiming_uniform_(E_r, a=math.sqrt(5))
        else:
            D_l, D_r = build_D(role_matrix, sparse=sparse)
            E_l, E_r = build_E(role_matrix, sparse=sparse)
        self.cons_only = cons_only
        if not self.cons_only:
            self.car_net = CarNet(D_l, tpr, sparse=sparse)
            self.cdr_net = CdrNet(D_r, tpr, sparse=sparse)
        root_role = None if sparse else role_matrix[0]
        self.cons_net = ConsNet(E_l, E_r, root_role, tpr, sparse=sparse)
        self.tpr = tpr
        self.num_ops = num_ops
        self.sparse = sparse
        self.filler_threshold = filler_threshold
        self.max_filled_roles = max_filled_roles
        self.max_roles_during_training = config.max_filled_roles
        self.max_roles_during_eval = config.tpr.num_roles

        self.new_tree_filler_dropout1d = None
        if new_tree_filler_dropout1d:
            self.new_tree_filler_dropout1d = nn.Dropout1d(p=new_tree_filler_dropout1d)

    def forward(self, memory, arg_weights, root_filler, op_dist, bsz, calculate_entropy=False, skip_dropout=False):
        car_arg_weights = arg_weights[:, :, 0]
        cdr_arg_weights = arg_weights[:, :, 1]
        cons_arg1_weights = arg_weights[:, :, 2]
        cons_arg2_weights = arg_weights[:, :, 3]

        # TODO: root_filler is very small, maybe we should normalize it? But why is it small in the first place? It's
        #  a weighted sum over the filler embeddings....
        if self.sparse:
            if type(memory) == torch.Tensor:
                memory = SparseTPRBlock(memory.indices(), memory.values())

            # TODO: large memory usage here
            cons_output = self.cons_net(
                memory, arg1_weight=cons_arg1_weights,
                arg2_weight=cons_arg2_weights, root_filler=root_filler,
                calculate_entropy=calculate_entropy
            )[0]
            if self.cons_only:
                output = cons_output
            else:
                car_output = self.car_net(
                    memory,
                    arg1_weight=car_arg_weights,
                    calculate_entropy=calculate_entropy
                )[0]
                cdr_output = self.cdr_net(
                    memory,
                    arg1_weight=cdr_arg_weights,
                    calculate_entropy=calculate_entropy
                )[0]

                indices = torch.stack(
                    (torch.cat((car_output.indices()[0], cdr_output.indices()[0], cons_output.indices()[0])),
                     torch.cat((car_output.indices()[1], cdr_output.indices()[1], cons_output.indices()[1])))
                )

                values = torch.cat(
                    (op_dist[:, 0][car_output.indices()[0]].unsqueeze(1) * car_output.values(),
                     op_dist[:, 1][cdr_output.indices()[0]].unsqueeze(1) * cdr_output.values(),
                     op_dist[:, 2][cons_output.indices()[0]].unsqueeze(1) * cons_output.values())
                )
                # TODO: figure out why shrinking the norm via op_dist isn't restored when the values are added together.
                #  For example, the norm of the filler in the root node will be less because it is multiplied by the
                #  attention weight, but then three values are added together which I would imagine restores the norm?
                #  Maybe not exactly, the norm of the summation involves cosines between the vectors.

                output = SparseTPR(*coalesce(indices, values))

            if self.new_tree_filler_dropout1d and not skip_dropout:
                values = self.new_tree_filler_dropout1d(output.values())
                mask = values.any(dim=-1)
                output = SparseTPR(output.indices()[:, mask], values[mask])

            # TODO: should we turn off max_filled_roles during evaluation? This leads to a dimension issue where the
            #  sparse memory expects max_filled_roles but at evaluation time we can have more than max_filled roles.
            if self.filler_threshold:
                threshold_mask = output.values().norm(dim=-1) > self.filler_threshold
                indices = output.indices()[:, threshold_mask]
                values = output.values()[threshold_mask]
                output = torch.sparse_coo_tensor(indices=indices,
                                                 values=values,
                                                 size=(bsz, self.tpr.num_roles, self.tpr.d_filler)).coalesce()
            elif self.max_filled_roles:
                batch_indices = torch.arange(bsz, device=output.device)[:, None]
                batch_indices_mask = output.indices()[0] == batch_indices
                batch_filler_norms = torch.where(batch_indices_mask, output.values().norm(dim=-1), -1)
                # We want to return at most max_filled_roles fillers per batch element, but we don't want k to be
                # more than the number of elements in output.values().norm(dim=-1).
                k = torch.clamp(batch_indices_mask.sum(dim=-1).max(), max=self.max_filled_roles)
                topk_values_and_indices = torch.topk(batch_filler_norms, k, sorted=False)
                topk_indices = topk_values_and_indices[1][topk_values_and_indices[0] != -1]
                output = torch.sparse_coo_tensor(indices=output.indices()[:, topk_indices],
                                                 values=output.values()[topk_indices],
                                                 size=(bsz, self.tpr.num_roles, self.tpr.d_filler)).coalesce()
            else:
                output = torch.sparse_coo_tensor(
                    indices=output.indices(),
                    values=output.values(),
                    size=(bsz, self.tpr.num_roles, self.tpr.d_filler)
                ).coalesce()
        else:
            output = op_dist[:, 2].view(bsz, 1, 1) * self.cons_net(
                memory, arg1_weight=cons_arg1_weights,
                arg2_weight=cons_arg2_weights,
                root_filler=root_filler,
                calculate_entropy=calculate_entropy
            )[0]
            if not self.cons_only:
                # We view op_dist as (bsz, 1, 1) so that it can be broadcast with the output (bsz, d_filler, d_role)
                output += op_dist[:, 0].view(bsz, 1, 1) * self.car_net(memory, arg1_weight=car_arg_weights,
                                                                      calculate_entropy=calculate_entropy)[0]
                output += op_dist[:, 1].view(bsz, 1, 1) * self.cdr_net(memory, arg1_weight=cdr_arg_weights,
                                                                       calculate_entropy=calculate_entropy)[0]
        return output


class CarNet(nn.Module):
    def __init__(self, D_0, tpr, sparse=False) -> None:
        super().__init__()
        # hardcoded op
        self.register_buffer('car_weight', D_0)
        self.tpr = tpr
        self.sparse = sparse

    def forward(self, x, arg1_weight, calculate_entropy=False):
        if calculate_entropy:
            arg1_entropy = torch.distributions.Categorical(arg1_weight).entropy() / np.log(arg1_weight.shape[-1]) if \
                arg1_weight.shape[-1] > 1 else torch.zeros(arg1_weight.shape[0], device=x.device)

        if self.sparse:
            # TODO: arg1_values take a lot of memory
            arg1_values = x.values() * arg1_weight[x.batch_indices(), x.memory_slot_indices()].unsqueeze(1)
            # TODO: probably a minor memory savings, but SparseTPR can take separate vectors for the indicies instead
            #  of expecting a matrix. This way I don't have to use stack to allocate more memory, I can reuse the
            #  underlying vectors.
            arg1 = SparseTPR(*coalesce(torch.stack((x.batch_indices(), x.role_indices())), arg1_values))
        else:
            # batch, length, filler, role x batch, length
            arg1 = torch.einsum('blfr,bl->bfr', x, arg1_weight)

        if self.tpr.empty_filler_initialization == 'random':
            # If the empty filler is not the 0 vector, we need to add the empty filler to the leaves which are zeroed
            # out by car_weight
            # batch, filler, role_from x role_[t]o, role_from
            output = torch.einsum('bfr,tr->bft', arg1, self.car_weight) + self.tpr.empty_leaves_tpr(device=x.device)
        else:
            if self.sparse:
                # we can perform car in a sparse way by using sparse.values() to get the fillers and filter for the ones
                # that survive the car operation. We can then use sparse.indices() to get the indices of the fillers
                # that survive and figure out what the new index is for each filler.
                roles = arg1.indices()[1]
                mask = roles % 2 == 0
                left_child_fillers = arg1.values()[mask]
                old_roles = roles[mask]
                new_roles = old_roles >> 1
                output = SparseTPR(torch.stack((arg1.indices()[0][mask], new_roles)), left_child_fillers)
            else:
                # batch, filler, role_from x role_[t]o, role_from
                output = torch.einsum('bfr,tr->bft', arg1, self.car_weight)

        return output, arg1_entropy if calculate_entropy else None, torch.max(arg1_weight, dim=1)[0]


class CdrNet(nn.Module):
    def __init__(self, D_1, tpr, sparse=False) -> None:
        super().__init__()
        # hardcoded op
        self.register_buffer('cdr_weight', D_1)
        self.tpr = tpr
        self.sparse = sparse

    def forward(self, x, arg1_weight, calculate_entropy=False):
        if calculate_entropy:
            arg1_entropy = torch.distributions.Categorical(arg1_weight).entropy() / np.log(arg1_weight.shape[-1]) if \
                arg1_weight.shape[-1] > 1 else torch.zeros(arg1_weight.shape[0], device=x.device)

        if self.sparse:
            arg1_values = x.values() * arg1_weight[x.batch_indices(), x.memory_slot_indices()].unsqueeze(1)
            arg1 = SparseTPR(*coalesce(torch.stack((x.batch_indices(), x.role_indices())), arg1_values))
        else:
            # batch, length, filler, role
            arg1 = torch.einsum('blfr,bl->bfr', x, arg1_weight)

        if self.tpr.empty_filler_initialization == 'random':
            # If the empty filler is not the 0 vector, we need to add the empty filler to the leaves which are zeroed
            # out by car_weight
            # batch, filler, role_from x role_[t]o, role_from
            output = torch.einsum('bfr,tr->bft', arg1, self.cdr_weight) + self.tpr.empty_leaves_tpr(device=x.device)
        else:
            if self.sparse:
                # we can perform cdr in a sparse way by using sparse.values() to get the fillers and filter for the ones
                # that survive the cdr operation. We can then use sparse.indices() to get the indices of the fillers
                # that survive and figure out what the new index is for each filler.
                roles = arg1.indices()[1]
                mask = torch.logical_and(roles != 1, roles % 2 == 1)
                right_child_fillers = arg1.values()[mask]
                old_roles = roles[mask]
                new_roles = old_roles >> 1
                output = SparseTPR(torch.stack((arg1.indices()[0][mask], new_roles)), right_child_fillers)
            else:
                # batch, filler, role_from x role_[t]o, role_from
                output = torch.einsum('bfr,tr->bft', arg1, self.cdr_weight)

        return output, arg1_entropy if calculate_entropy else None, torch.max(arg1_weight, dim=1)[0]


class ConsNet(nn.Module):
    def __init__(self, E_0, E_1, root_role, tpr, sparse=False) -> None:
        super().__init__()
        # hardcoded op
        self.register_buffer('cons_l', E_0)
        self.register_buffer('cons_r', E_1)
        self.root_role = root_role
        self.sparse = sparse
        self.tpr = tpr

    def forward(self, x, arg1_weight, arg2_weight, root_filler, calculate_entropy=False):
        if calculate_entropy:
            arg1_entropy = torch.distributions.Categorical(arg1_weight).entropy() / np.log(arg1_weight.shape[-1]) if \
                arg1_weight.shape[-1] > 1 else torch.zeros(arg1_weight.shape[0], device=x.device)
            arg2_entropy = torch.distributions.Categorical(arg2_weight).entropy() / np.log(arg2_weight.shape[-1]) if \
                arg2_weight.shape[-1] > 1 else torch.zeros(arg2_weight.shape[0], device=x.device)

        if self.sparse:
            pad_mask = (x.values() == 0).all(dim=1)

            arg1_weight_expanded = arg1_weight[x.batch_indices(), x.memory_slot_indices()].unsqueeze(1)
            arg1_weight_expanded[pad_mask] = 0
            arg1_values = x.values() * arg1_weight_expanded
            arg1 = SparseTPR(*coalesce(torch.stack((x.batch_indices(), x.role_indices())), arg1_values))

            arg2_weight_expanded = arg2_weight[x.batch_indices(), x.memory_slot_indices()].unsqueeze(1)
            arg2_weight_expanded[pad_mask] = 0
            arg2_values = x.values() * arg2_weight_expanded
            arg2 = SparseTPR(*coalesce(torch.stack((x.batch_indices(), x.role_indices())), arg2_values))
        else:
            # batch, length, filler, role
            arg1 = torch.einsum('blfr,bl->bfr', x, arg1_weight)
            arg2 = torch.einsum('blfr,bl->bfr', x, arg2_weight)

        if self.sparse:
            arg1_roles = arg1.indices()[1]
            # Mask out nodes that are already at the bottom level, they don't survive cons
            arg1_mask = arg1_roles < self.tpr.max_interior_index
            arg1_fillers = arg1.values()[arg1_mask]
            arg1_old_roles = arg1_roles[arg1_mask]
            arg1_new_roles = arg1_old_roles << 1

            arg2_roles = arg2.indices()[1]
            # Mask out nodes that are already at the bottom level, they don't survive cons
            arg2_mask = arg2_roles < self.tpr.max_interior_index
            arg2_fillers = arg2.values()[arg2_mask]
            arg2_old_roles = arg2_roles[arg2_mask]
            arg2_new_roles = (arg2_old_roles << 1) + 1

            root_batch_indices = torch.arange(root_filler.shape[0], device=arg1_weight.device)
            root_roles = torch.ones(root_filler.shape[0], dtype=torch.int32, device=arg1_weight.device)

            new_roles = torch.cat((root_roles, arg1_new_roles, arg2_new_roles))
            new_fillers = torch.cat((root_filler, arg1_fillers, arg2_fillers))
            batch_indices = torch.cat((root_batch_indices, arg1.indices()[0][arg1_mask], arg2.indices()[0][arg2_mask]))

            output = SparseTPR(torch.stack((batch_indices, new_roles)), new_fillers)
        else:
            output = F.linear(arg1, self.cons_l) + F.linear(arg2, self.cons_r) + torch.einsum('bf,r->bfr', root_filler,
                                                                                              self.root_role)

        return output, arg1_entropy if calculate_entropy else None, arg2_entropy if calculate_entropy else None, \
            torch.max(arg1_weight, dim=1)[0], torch.max(arg2_weight, dim=1)[0]


class MemorySet(torch.autograd.Function):
    @staticmethod
    def forward(ctx, memory, x, start_index):
        # If x.dim() == 3, we are storing a single TPR. If x.dim() == 4, we are storing a block of TPRs.
        if x.dim() == 3:
            memory.data[:, start_index].copy_(x)
            ctx.start_index = start_index
        else:
            end_index = start_index + x.shape[1]
            memory.data[:, start_index:end_index].copy_(x)
            ctx.start_index = start_index
            ctx.end_index = end_index

        return memory

    @staticmethod
    def backward(ctx, grad_out):
        start_index = getattr(ctx, 'start_index')
        end_index = getattr(ctx, 'end_index', None)
        if end_index:
            return grad_out, grad_out[:, start_index:end_index], None
        else:
            return grad_out, grad_out[:, start_index], None


class SparseStorageSet(torch.autograd.Function):
    @staticmethod
    def forward(ctx, indices_storage, values_storage, x_indices, x_values, storage_index,
                memory_value, storage_start_end_indices):
        filled_roles = x_values.shape[0]

        start_index = storage_start_end_indices[storage_index]
        end_index = start_index + filled_roles
        storage_start_end_indices[storage_index + 1] = end_index

        values_storage.data[start_index:end_index].copy_(x_values)
        # For some reason, I can't write both the batch and role indices of x_indices into indices_memory in a single
        # command, so write them in two commands instead
        indices_storage.data[0, start_index:end_index].copy_(x_indices[0])
        indices_storage.data[1, start_index:end_index].copy_(memory_value)
        indices_storage.data[2, start_index:end_index].copy_(x_indices[1])

        ctx.start_index = start_index
        ctx.end_index = end_index
        return indices_storage, values_storage

    @staticmethod
    def backward(ctx, indices_grad, values_grad):
        return None, values_grad, None, values_grad[ctx.start_index:ctx.end_index], None, None, None


sparse_storage_set = SparseStorageSet.apply


class SparseStorageBlockSet(torch.autograd.Function):
    @staticmethod
    def forward(
            ctx,
            indices_storage: Tensor,
            values_storage: Tensor,
            x_indices,
            x_values,
            storage_index,
            storage_start_end_indices
    ):
        filled_roles = x_values.shape[0]

        start_index = storage_start_end_indices[storage_index]
        end_index = start_index + filled_roles
        storage_start_end_indices[storage_index + 1] = end_index

        values_storage.data[start_index:end_index].copy_(x_values)
        # For some reason, I can't write both the batch, memory, and role indices of x_indices into indices_memory in a
        # single command, so write them in separate commands instead
        indices_storage.data[0, start_index:end_index].copy_(x_indices[0])
        indices_storage.data[1, start_index:end_index].copy_(x_indices[1])
        indices_storage.data[2, start_index:end_index].copy_(x_indices[2])

        ctx.start_index = start_index
        ctx.end_index = end_index
        return indices_storage, values_storage

    @staticmethod
    def backward(ctx, indices_grad, values_grad):
        return None, values_grad, None, values_grad[ctx.start_index:ctx.end_index], None, None


sparse_storage_block_set = SparseStorageBlockSet.apply


def get_sparse_tpr_from_storage(
        indices_storage: Tensor,
        values_storage: Tensor,
        storage_index: int,
        storage_start_end_indices: Tensor
) -> SparseTPR:
    """
    This returns a single SparseTPR from memory. If you want a block of TPRs from memory, use
    `get_sparse_tpr_block_from_storage()` instead.
    """
    start_index = storage_start_end_indices[storage_index]
    end_index = storage_start_end_indices[storage_index + 1]

    indices = indices_storage[[0, 2], start_index:end_index]
    return SparseTPR(indices, values_storage[start_index:end_index])


def get_sparse_tpr_block_from_storage(
        indices_storage: Tensor,
        values_storage: Tensor,
        storage_end_index: int,
        storage_start_end_indices: Tensor,
        storage_start_index: int = 0,
) -> SparseTPRBlock:
    """
    This returns a SparseTPRBlock from memory. The block starts from storage_start_end_indices[storage_start_index]
    and ends at storage_start_end_indices[memory_end_index]. If you want a
    single TPR from memory, use `get_sparse_tpr_from_memory()` instead.
    """
    start_index = storage_start_end_indices[storage_start_index]
    end_index = storage_start_end_indices[storage_end_index]

    return SparseTPRBlock(indices_storage[:, start_index:end_index], values_storage[start_index:end_index])


memory_set = MemorySet.apply


class SetTransformer(nn.Module):
    # TODO: this isn't really a SetTransformer, it's just pooling by multiheaded attention plus our tree position
    #  scheme. We should refactor it. But also, maybe using a full SetTransformer is a good idea.
    def __init__(self, d_model, d_filler, role_bits, nhead, dim_feedforward):
        super().__init__()
        d_input = int(d_filler + role_bits)
        # binary_mask is used to convert the role integer to a binary vector representation
        binary_mask = 2 ** torch.arange(role_bits)
        self.register_buffer('binary_mask', binary_mask)
        self.d_key = d_model // nhead
        self.d_model = d_model
        self.nhead = nhead
        self.query_vectors = nn.Parameter(torch.empty(nhead, self.d_key))
        nn.init.xavier_uniform_(self.query_vectors)
        # I'm not sure if bias helps or not, but since we have many implicit 0 vectors, a bias of 0 means that the
        # linear transformation still produces 0 so this maintains sparsity.
        # TODO: how should I initialize these weights?
        self.proj_k = nn.Linear(d_input, d_model, bias=False)
        self.proj_v = nn.Linear(d_input, d_model, bias=False)
        self.proj_o = nn.Linear(d_model, d_model)
        self.mha_layer_norm = nn.LayerNorm(d_input)
        self.ff_layer_norm = nn.LayerNorm(d_model)
        self.output_layer_norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, d_model)
        )
        self.input_filler_norm = nn.LayerNorm(d_filler)

    def forward(self, tree_to_shrink: SparseTPRBlock, bsz: int, num_trees: int):
        tree_position_dimensions = self._generate_tree_positional_encodings(tree_to_shrink)
        query_key_match, value = self._attention(tree_to_shrink, tree_position_dimensions)

        num_nodes = value.shape[0]
        # We need to use masks so that softmax only normalizes over elements in the same batch and memory slot
        # We want a mask that is num_trees x num_nodes where each row indicates the nodes which belong to that tree
        #  The nodes which belong to a tree is a combination of the batch index and memory index.
        batch_index = tree_to_shrink.batch_indices()
        batch_mask = torch.arange(bsz, device=batch_index.device).unsqueeze(0)
        batch_mask = batch_index.unsqueeze(1) == batch_mask

        memory_slots = tree_to_shrink.memory_slot_indices().max() + 1
        memory_index = tree_to_shrink.memory_slot_indices()
        memory_mask = torch.arange(memory_slots, device=batch_index.device).unsqueeze(0)
        memory_mask = memory_index.unsqueeze(1) == memory_mask

        batch_and_memory_mask = torch.logical_and(batch_mask.unsqueeze(-1).expand(-1, -1, memory_slots),
                                                  memory_mask.unsqueeze(1).expand(-1, bsz, -1)).view(num_nodes, -1)

        # This mask will be used by the agent to pad out the tokens that should be ignored
        agent_pad_mask = ~batch_and_memory_mask.any(dim=0).view(bsz, -1)
        batch_and_memory_mask = batch_and_memory_mask[:, batch_and_memory_mask.any(dim=0)]

        # The mask should be shared across all heads
        batch_and_memory_mask = batch_and_memory_mask.unsqueeze(-1).expand(-1, -1, query_key_match.shape[-1])
        query_key_match = query_key_match.unsqueeze(1).expand(-1, num_trees, -1)
        query_key_match_masked = torch.where(batch_and_memory_mask, query_key_match, -1e9)

        attention = F.softmax(query_key_match_masked / np.sqrt(self.query_vectors.shape[1]), dim=0)
        # [n]odes x [h]eads x [v]alue, [n]odes x [t]rees x [h]eads -> [t]rees x [h]eads x [v]alue
        out = torch.einsum('nhv,nth->thv', value, attention)

        return (self._post_attention_computation(out.view(num_trees, -1), bsz, memory_slots, agent_pad_mask),
                agent_pad_mask)

    def single_tree_forward(self, tree_to_shrink: SparseTPR, bsz: int):
        tree_position_dimensions = self._generate_tree_positional_encodings(tree_to_shrink)
        query_key_match, value = self._attention(tree_to_shrink, tree_position_dimensions)

        # We need to use masks so that softmax only normalizes over elements in the same batch
        batch_index = tree_to_shrink.indices()[0]
        mask = torch.arange(bsz, device=batch_index.device).unsqueeze(1)
        mask = batch_index.unsqueeze(0) == mask

        # TODO: can I leverage FlashAttention for memory savings here? It might be difficult because of the masking
        query_key_match = query_key_match.unsqueeze(0).repeat(bsz, 1, 1)
        mask_expanded = mask.unsqueeze(-1).expand(-1, -1, query_key_match.shape[-1])
        query_key_match_masked = torch.where(mask_expanded, query_key_match, -1e9)
        attention = F.softmax(query_key_match_masked / np.sqrt(self.query_vectors.shape[1]), dim=1)

        out = torch.einsum('ehv,beh->bhv', value, attention)

        # Each batch gets a single tree back that should not be masked
        agent_pad_mask = torch.tensor([False], device=out.device).expand(bsz, -1)
        return (self._post_attention_computation(out.view(bsz, -1), bsz, memory_slots=1, agent_pad_mask=agent_pad_mask),
                agent_pad_mask)

    def _attention(self, tree_to_shrink, tree_position_dimensions):
        num_nodes = tree_position_dimensions.shape[0]
        # TODO: this line uses a decent amount of memory
        # TODO: I think our fillers keep shrinking, we need a way to deal with that (although maybe it's just the new
        #  fillers that are bringing the average down)
        filler_and_role_vectors = torch.cat(
            (self.input_filler_norm(tree_to_shrink.values()), tree_position_dimensions),
            1
        )
        # Pre-norm the input
        filler_and_role_vectors = self.mha_layer_norm(filler_and_role_vectors)
        key = self.proj_k(filler_and_role_vectors).view(num_nodes, self.nhead, self.d_key)
        value = self.proj_v(filler_and_role_vectors).view(num_nodes, self.nhead, self.d_key)

        # TODO: make this line more efficient
        query_key_match = torch.bmm(
            self.query_vectors.unsqueeze(1),
            key.transpose(0, 1).transpose(1, 2)
        ).squeeze(1).transpose(0, 1)
        return query_key_match, value

    def _post_attention_computation(self, weighted_value, bsz, memory_slots, agent_pad_mask):
        attention_out = self.proj_o(weighted_value)

        # Residual connection
        residual_one = attention_out + self.query_vectors.view(-1)
        layer_norm_one = self.ff_layer_norm(residual_one)
        ff_output = self.ff(layer_norm_one)

        # Residual connection
        residual_two = ff_output + residual_one
        layer_norm_two = self.output_layer_norm(residual_two)

        rectangle_out = torch.zeros((bsz, memory_slots, self.d_model), device=layer_norm_two.device)
        rectangle_out[~agent_pad_mask] = layer_norm_two

        return rectangle_out

    def _generate_tree_positional_encodings(self, tree_to_shrink: SparseTPR):
        """

        """
        if type(tree_to_shrink) == torch.Tensor:
            tree_to_shrink = SparseTPR(tree_to_shrink.indices(), tree_to_shrink.values())

        tree_position_dimensions = \
            tree_to_shrink.role_indices().unsqueeze(-1).bitwise_and(self.binary_mask).ne(0).float()

        # The tree position dimensions are binary vectors with a 1 at the rightmost element in order to differentiate
        # the left child from the left-left child. We can remove that rightmost 1 here, and then normalize the values
        # by converting 0s to -1. We only need to consider the values to the left of the rightmost 1, since the 0s to
        # the right of that rightmost 1 are empty and don't signify a left branch.

        # Reverse the rows to find the rightmost '1' using argmax
        reversed_matrix = torch.flip(tree_position_dimensions, [1])
        # Find the indices of non-zero elements (in this case, '1's)
        max_non_zero_indices = torch.argmax(reversed_matrix, dim=1, keepdim=True)

        # Correct indices since the matrix was reversed
        corrected_max_non_zero_indices = tree_position_dimensions.size(1) - 1 - max_non_zero_indices

        arange = torch.arange(tree_position_dimensions.size(1), device=tree_position_dimensions.device).unsqueeze(0)
        branch_mask = (arange < corrected_max_non_zero_indices)

        # Zero out the rightmost ones
        tree_position_dimensions.scatter_(1, corrected_max_non_zero_indices, 0)

        # Subtract 0.5 and multiply by 2 where mask is True
        # TODO: What should the scale of the position embeddings be?
        tree_position_dimensions[branch_mask] = (tree_position_dimensions[branch_mask] - 0.5) * 2
        return tree_position_dimensions


class SumModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.sum(self.dim)


def arg_weight_formatter(x):
    formatted = "{:.2f}".format(x)
    if formatted.startswith('1.00'):
        formatted = '1.0'
    else:
        formatted = formatted[1:]
    return formatted


'''
indices = list(range(1, 8))
index = 3
mask = int('0b' + '1' * (index.bit_length() - 1), 2) & index

children will have an index greater than the parent
the least significant x bits will match where x is the depth of the parent

parent XOR child == 0 for the first x bits. The difficulty is that if the parent is 000, it will match with 0 which is
actually it's grandparent. So maybe I should just do something like (parent XOR child == 0) AND child > parent for the first x-1 bits without
first removing the msb from everyone. I'll still need clz/bit_length to find out what x is.

remember for the above parent & child, I need to remove the msb 1
'''
