import logging
import os
import time
from typing import Dict, List

import torch
import torch.nn as nn
import wandb
from nltk.tree import Tree, TreePrettyPrinter
from torch.utils.data import DataLoader

from TPR_utils import TPR, batch_symbols_to_node_tree, SparseTPR
from models import RootPredictionType

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        tpr: TPR,
        data_loaders: Dict[str, DataLoader],
        optimizer: torch.optim.Optimizer,
        num_epochs: int,
        num_steps: int,
        num_warmup_steps: int,
        main_process: bool,
        is_ddp: bool,
        decoded_tpr_to_tree_fn: callable,
        xent_loss: nn.CrossEntropyLoss,
        device: str,
        output_index2vocab: List[str],
        vocab_info: Dict[str, tuple],
        use_wandb: bool = True,
        validate_every_num_epochs: int = 1,
        train_log_freq: int = 20,
        early_stop_epochs: int = 5,
        pad_idx: int = 0,
        sparse: bool = True,
        scheduler: torch.optim.lr_scheduler.LRScheduler = None,
        gclip: float = 1.,
        lr: float = 1e-4,
        out_dir: str = 'out',
        best_checkpoint_file: str = 'best_checkpoint.pt',
        most_recent_checkpoint_file: str = 'most_recent_checkpoint.pt',
        use_custom_memory: bool = False,
        cross_entropy_weighting: str = None,
        entropy_regularization_coefficient: float = 0.,
        max_input_length: int = 0,
        nt_token_index: int = 0,
        eob_token_index: int = None,
        output_indices_mask: List[int] = None,
    ):
        self.model = model
        self.tpr = tpr
        self.train_loader = data_loaders.get('train')
        self.val_loader = data_loaders.get('valid')
        self.data_loaders = data_loaders
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.num_steps = num_steps
        self.num_warmup_steps = num_warmup_steps
        self.main_process = main_process
        self.is_ddp = is_ddp
        self.decoded_tpr_to_tree_fn = decoded_tpr_to_tree_fn
        self.xent_loss = xent_loss
        self.device = device
        self.index2vocab = output_index2vocab
        self.vocab_info = vocab_info
        self.use_wandb = use_wandb
        self.validate_every_num_epochs = validate_every_num_epochs
        self.train_log_freq = train_log_freq
        self.early_stop_epochs = early_stop_epochs
        self.pad_idx = pad_idx
        self.sparse = sparse
        self.scheduler = scheduler
        self.gclip = gclip
        self.lr = lr
        self.out_dir = out_dir
        self.best_checkpoint_file = best_checkpoint_file
        self.most_recent_checkpoint_file = most_recent_checkpoint_file
        self.use_custom_memory = use_custom_memory
        self.cross_entropy_weighting = cross_entropy_weighting
        self.entropy_regularization_coefficient = entropy_regularization_coefficient
        self.max_input_length = max_input_length

        self.global_step = 0
        self.best_val_loss = float('inf')
        self.early_stop_not_improving_counter = 0
        self.epoch = 0
        self.best_val_full_accuracy = 0
        self.early_stop_perfect_val_acc_counter = 0
        if is_ddp:
            self.raw_model = model.module
        else:
            self.raw_model = model
        self.nt_token_index = nt_token_index
        self.eob_token_index = eob_token_index
        self.output_indices_mask = torch.tensor(output_indices_mask, device=device)

    def train(self):
        print('Start training')
        stop_training = False
        best_train_acc = 0
        while self.epoch < self.num_epochs:
            is_best_epoch = False
            if stop_training:
                break

            epoch_start_time = time.time()
            # Note: accuracy and loss are just estimates from the last set of batches in the epoch
            train_acc, train_partial_acc, train_loss, lr = self.train_epoch()
            if self.main_process and (self.epoch % self.validate_every_num_epochs == 0
                                      or (self.num_steps and self.global_step == self.num_steps)):
                val_loss, val_partial_accuracy, val_full_accuracy, val_entropies, val_perplexity = self.evaluate()
                if val_loss <= self.best_val_loss:
                    self.best_val_loss = val_loss
                    is_best_epoch = True

                checkpoint = {
                    'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    # 'args': args,
                    'step': self.global_step,
                    'best_valid_loss': self.best_val_loss,
                    'epoch': self.epoch,
                    'wandb_id': wandb.run.id
                }
                print(f'saving most recent checkpoint to'
                      f' {os.path.join(self.out_dir, self.most_recent_checkpoint_file)}')
                torch.save(checkpoint, os.path.join(self.out_dir, self.most_recent_checkpoint_file))

                if is_best_epoch:
                    print(f'saving best checkpoint to {os.path.join(self.out_dir, self.best_checkpoint_file)}')
                    torch.save(checkpoint, os.path.join(self.out_dir, self.best_checkpoint_file))
                    self.early_stop_not_improving_counter = 0
                else:
                    self.early_stop_not_improving_counter += 1

                if self.early_stop_not_improving_counter == self.early_stop_epochs:
                    print(
                        'Validation loss did not improve for {} epochs, stopping early'.format(self.early_stop_epochs))
                    stop_training = True

                if val_full_accuracy >= self.best_val_full_accuracy:
                    self.best_val_full_accuracy = val_full_accuracy

                # TODO: How does early stopping work with the other processes? Do we need to do something special to
                #  make sure that the other processes stop?
                if val_full_accuracy == 1.0:
                    self.early_stop_perfect_val_acc_counter += 1
                else:
                    self.early_stop_perfect_val_acc_counter = 0

                # TODO: this number should be a commandline arg
                if self.early_stop_perfect_val_acc_counter == 100:
                    print('Validation accuracy reached 100% for 100 epochs, stopping early')
                    stop_training = True

                epoch_end_time = time.time()
                epoch_elapsed = epoch_end_time - epoch_start_time
                train_rate = len(self.train_loader.dataset) / epoch_elapsed
                print(f'epoch: {self.epoch:,}')

                print(
                    f'  Train Acc: {train_acc:.2f}, partial_train_acc: {train_partial_acc:.2f}, total_loss:'
                    f' {train_loss:.5f}, lr: {lr:.10f}, '
                    f'samples/sec: {train_rate:.2f}, time for epoch: {epoch_elapsed:.2f}s')
                print(
                    f'  Valid Acc: {val_full_accuracy:.2f}, partial_valid_acc: {val_partial_accuracy:.2f}, valid_loss:'
                    f' {val_loss:.5f}, valid_perplexity: {val_perplexity: .2f}')

                if self.main_process and self.use_wandb:
                    cons_arg1_dict = {f'Val arg 1 step {idx}': value.item() for idx, value in
                                      enumerate(val_entropies['cons_arg1'])}
                    cons_arg2_dict = {f'Val arg 2 step {idx}': value.item() for idx, value in
                                      enumerate(val_entropies['cons_arg2'])}
                    wandb.log(dict(
                        {**cons_arg1_dict, **cons_arg2_dict},
                        epoch=self.epoch,
                        valid_acc=val_full_accuracy,
                        valid_loss=val_loss,
                        valid_partial_acc=val_partial_accuracy,
                        valid_perplexity=val_perplexity,
                    ), step=self.global_step)

                if self.main_process:
                    for test_set in ['test', 'eval_long', 'eval_new', 'eval_illformed']:
                        if self.data_loaders.get(test_set):
                            loss, partial_acc, full_acc, test_entropies, test_perplexity = self.test(
                                self.data_loaders.get(test_set))
                            logger.info(
                                f'{test_set}\t  full_acc: {full_acc:.2f}, partial_acc: {partial_acc:.2f}, loss: {loss:.5f}'
                            )
                            if self.use_wandb:
                                #cons_arg1_dict = {f'Test arg 1 step {idx}': value.item() for idx, value in
                                #                  enumerate(test_entropies['cons_arg1'])}
                                #cons_arg2_dict = {f'Test arg 2 step {idx}': value.item() for idx, value in
                                #                  enumerate(test_entropies['cons_arg2'])}

                                test_dict = {
                                        f'{test_set}_loss': loss,
                                        f'{test_set}_partial_acc': partial_acc,
                                        f'{test_set}_full_acc': full_acc,
                                        f'{test_set}_perplexity': test_perplexity
                                    }
                                #test_dict.update(cons_arg1_dict)
                                #test_dict.update(cons_arg2_dict)
                                wandb.log(
                                    test_dict, step=self.global_step
                                )
            best_train_acc = max(train_acc, best_train_acc)
            if self.is_ddp:
                torch.distributed.barrier()
        print('Finished training')

    def train_epoch(self):
        self.model.train()
        loss_accumulator = torch.tensor(0., device=self.device)
        correct_tokens_accumulator = torch.tensor(0., device=self.device)
        total_tokens_accumulator = torch.tensor(0., device=self.device)
        correct_sequences_accumulator = torch.tensor(0., device=self.device)
        total_sequences_accumulator = torch.tensor(0., device=self.device)
        accumulator_steps = torch.tensor(0., device=self.device)
        cons_arg1_entropy_accumulator = torch.zeros(self.model.dtm_layers, device=self.device)
        cons_arg2_entropy_accumulator = torch.zeros(self.model.dtm_layers, device=self.device)
        start_time = time.time()

        def reset_accumulators():
            loss_accumulator.zero_()
            correct_tokens_accumulator.zero_()
            total_tokens_accumulator.zero_()
            correct_sequences_accumulator.zero_()
            total_sequences_accumulator.zero_()
            accumulator_steps.zero_()
            cons_arg1_entropy_accumulator.zero_()
            cons_arg2_entropy_accumulator.zero_()

        for epoch_step, batch in enumerate(self.train_loader):
            #print(epoch_step)
            #logger.verbose(f'Epoch {self.epoch}, step {epoch_step}/{len(self.train_loader)}')
            # linearly increase the learning rate if we are in the warmup period
            is_warmup = self.global_step < self.num_warmup_steps
            if is_warmup:
                lr = self.lr * self.global_step / self.num_warmup_steps
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
            else:
                lr = self.lr

            if (self.raw_model.nta.op_dist_fn == 'gumbel' or self.raw_model.nta.arg_dist_fn == 'gumbel') \
                    and gumbel_temp > .5:
                gumbel_temp = max(.5, 1 - 1 / self.num_steps * self.global_step)
                self.model.set_gumbel_temp(gumbel_temp)
                # print('Gumbel temp:', gumbel_temp)
            profile_model = False
            if profile_model:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
            batch_loss, batch_correct_tokens, batch_token_total, batch_correct_sequences, batch_total_sequences, _, \
                batch_entropies, _, _= self.process_batch(batch, use_custom_memory=self.use_custom_memory)
            batch_correct_sequences = batch_correct_sequences.sum()
            if profile_model:
                end.record()
                # Waits for everything to finish running
                torch.cuda.synchronize()
                logger.debug(f'Batch {epoch_step} forward took {start.elapsed_time(end)} ms and allocated '
                             f'{torch.cuda.memory_allocated() / 1024 ** 2} mb')

            if self.global_step < 10000:
                current_entropy_coef = self.entropy_regularization_coefficient * (self.global_step / 10000)
            else:
                current_entropy_coef = self.entropy_regularization_coefficient
            batch_loss += current_entropy_coef * (batch_entropies['cons_arg1'].mean() + batch_entropies['cons_arg2'].mean())

            loss_accumulator += batch_loss.detach()
            correct_tokens_accumulator += batch_correct_tokens.detach()
            total_tokens_accumulator += batch_token_total
            correct_sequences_accumulator += batch_correct_sequences.detach()
            total_sequences_accumulator += batch_total_sequences
            cons_arg1_entropy_accumulator += batch_entropies['cons_arg1'].detach()
            cons_arg2_entropy_accumulator += batch_entropies['cons_arg2'].detach()
            accumulator_steps += 1

            if profile_model:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
            batch_loss.backward()
            if profile_model:
                end.record()
                # Waits for everything to finish running
                torch.cuda.synchronize()
                logger.debug(f'Batch {epoch_step} backward took {start.elapsed_time(end)} ms and allocated '
                             f'{torch.cuda.memory_allocated() / 1024 ** 2} mb')
            # TODO: look into the norm of our gradients, it seems very small
            if self.gclip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gclip)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            if self.scheduler and not is_warmup:
                # adjust LR as per scheduler
                self.scheduler.step()
                lr = self.optimizer.param_groups[0]['lr']

            if self.train_log_freq != -1 and self.global_step % self.train_log_freq == 0 and self.main_process:
                train_acc = correct_sequences_accumulator / total_sequences_accumulator
                train_partial_acc = correct_tokens_accumulator / total_tokens_accumulator
                train_loss = loss_accumulator / accumulator_steps
                cons_arg1_entropy = cons_arg1_entropy_accumulator / accumulator_steps
                cons_arg2_entropy = cons_arg2_entropy_accumulator / accumulator_steps
                dt = time.time() - start_time
                start_time = time.time()
                logger.info(f'Epoch {self.epoch}, step {epoch_step}/{len(self.train_loader)}, '
                            f'train_acc: {train_acc:.2f}, train_partial_acc: {train_partial_acc:.2f}, '
                            f'train_loss: {train_loss:.3f}, time: {dt:.2f} s')
                #logger.info(f'Maximum GPU memory allocated: '
                #            f'{torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024)} gb')
                cons_arg1_dict = {f'Train arg 1 step {idx}': value.item() for idx, value in enumerate(cons_arg1_entropy)}
                cons_arg2_dict = {f'Train arg 2 step {idx}': value.item() for idx, value in enumerate(cons_arg2_entropy)}
                wandb.log(dict(
                    {**cons_arg1_dict, **cons_arg2_dict},
                    epoch=self.epoch,
                    train_acc=train_acc,
                    train_partial_acc=train_partial_acc,
                    train_loss=train_loss,
                    lr=lr,
                    # roles_filled=fully_decoded.values().numel() / bsz
                ), step=self.global_step)
                reset_accumulators()

            self.global_step += 1
            if self.global_step >= self.num_steps:
                break
        self.epoch += 1
        return (correct_sequences_accumulator / total_sequences_accumulator,
                correct_tokens_accumulator / total_tokens_accumulator, loss_accumulator / accumulator_steps, lr)

    def evaluate(self):
        return self._evaluate(self.val_loader, debug=True)

    def _evaluate(self, loader: DataLoader, debug: bool = False, print_incorrect_filename: str = None):
        print_incorrect_file = False
        if print_incorrect_filename:
            print_incorrect_file = open(print_incorrect_filename, 'w')
        with torch.inference_mode():
            self.model.eval()
            loss_accumulator = torch.tensor(0., device=self.device)
            correct_tokens_accumulator = torch.tensor(0., device=self.device)
            total_tokens_accumulator = torch.tensor(0., device=self.device)
            correct_sequences_accumulator = torch.tensor(0., device=self.device)
            total_sequences_accumulator = torch.tensor(0., device=self.device)
            cons_arg1_entropy_accumulator = torch.zeros(self.model.dtm_layers, device=self.device)
            cons_arg2_entropy_accumulator = torch.zeros(self.model.dtm_layers, device=self.device)
            perplexity_accumulator = torch.tensor(0., device=self.device)

            for local_step, batch in enumerate(loader):
                is_debug_step = local_step == 0
                (batch_loss, batch_correct_tokens, batch_token_total, batch_correct_sequences, batch_total_sequences,
                 debug_info, batch_entropies, _, batch_perplexity) = self.process_batch(
                    batch,
                    debug=is_debug_step and debug,
                    use_ddp_module=self.is_ddp,
                    use_custom_memory=self.use_custom_memory
                )
                if print_incorrect_file:
                    # This method is serial and slow, although it only happens once at the end of training so
                    # probably okay
                    for i, correct in enumerate(batch_correct_sequences):
                        if not correct:
                            print_incorrect_file.write(f'{batch["raw_input"][i]}\t{batch["raw_output"][i]}\n')

                batch_correct_sequences = batch_correct_sequences.sum()

                loss_accumulator += batch_loss.detach()
                correct_tokens_accumulator += batch_correct_tokens.detach().sum()
                total_tokens_accumulator += batch_token_total
                correct_sequences_accumulator += batch_correct_sequences.detach().sum()
                total_sequences_accumulator += batch_total_sequences
                cons_arg1_entropy_accumulator += batch_entropies['cons_arg1'].detach()
                cons_arg2_entropy_accumulator += batch_entropies['cons_arg2'].detach()
                perplexity_accumulator += batch_perplexity.detach()

                if is_debug_step and debug:
                    output_filler_indices = batch['output_fillers']
                    batch_size = output_filler_indices.shape[0]
                    output_role_indices = batch['output_roles']
                    output_batch_indices = torch.nonzero(output_role_indices, as_tuple=True)[0]
                    output_filler_indices = output_filler_indices[output_role_indices != 0]
                    output_role_indices = output_role_indices[output_role_indices != 0]
                    target = torch.sparse_coo_tensor(indices=torch.stack((output_batch_indices, output_role_indices)),
                                                     values=output_filler_indices, size=(batch_size,
                                                                                         self.tpr.num_roles)).coalesce()

                    formatted_tree = TreePrettyPrinter(Tree.fromstring(
                        batch_symbols_to_node_tree(
                            SparseTPR(target.indices(), target.values()),
                            self.index2vocab,
                            terminal_vocab=self.vocab_info['terminal'],
                            unary_vocab=self.vocab_info['unary'],
                            sparse=True
                        )[0].str()
                    ))
                    print('Correct output:\n{}'.format(formatted_tree.text()))

            if print_incorrect_file:
                print_incorrect_file.close()
                if self.use_wandb:
                    wandb.save(print_incorrect_filename)
            loss = loss_accumulator / len(loader)
            cons_arg1_entropy = cons_arg1_entropy_accumulator / len(loader)
            cons_arg2_entropy = cons_arg2_entropy_accumulator / len(loader)
            # TODO: should perplexity be averaged by the number of batches?
            perplexity = perplexity_accumulator / len(loader)
            partial_accuracy = correct_tokens_accumulator / total_tokens_accumulator
            full_accuracy = correct_sequences_accumulator / total_sequences_accumulator
            return loss, partial_accuracy, full_accuracy, {'cons_arg1': cons_arg1_entropy, 'cons_arg2':
                cons_arg2_entropy}, perplexity

    def test(self, test_loader: DataLoader, print_incorrect_filename: str = None):
        return self._evaluate(test_loader, print_incorrect_filename=print_incorrect_filename)

    def process_batch(self, batch: Dict[str, torch.Tensor], debug: bool = False, use_ddp_module: bool = False,
                      use_custom_memory: bool = False,):
        if len(batch['input_fillers'].shape) == 2:
            bsz, _ = batch['input_fillers'].shape
        else:
            bsz, len_, _ = batch['input_fillers'].shape

        # Add in the memory dimension if it doesn't exist
        if batch['input_fillers'].dim() == 2:
            batch['input_fillers'] = batch['input_fillers'].unsqueeze(1)
            batch['input_roles'] = batch['input_roles'].unsqueeze(1)
        # TODO: is it better to move these tensors to GPU here, or should I move input_ and target once the sparse
        #  tensors are created?
        input_filler_indices = batch['input_fillers'].to(self.device, non_blocking=True)
        input_role_indices = batch['input_roles'].to(self.device, non_blocking=True)
        output_filler_indices = batch['output_fillers'].to(self.device, non_blocking=True)
        output_role_indices = batch['output_roles'].to(self.device, non_blocking=True)

        input_batch_indices = torch.nonzero(input_role_indices, as_tuple=True)[0]
        input_memory_indices = torch.nonzero(input_role_indices, as_tuple=True)[1]
        input_filler_indices = input_filler_indices[input_role_indices != 0]
        input_role_indices = input_role_indices[input_role_indices != 0]

        input_ = torch.sparse_coo_tensor(
            indices=torch.stack((input_batch_indices, input_memory_indices, input_role_indices)),
            values=input_filler_indices,
            size=(bsz, batch['input_fillers'].shape[1], self.tpr.d_role)
        )

        output_batch_indices = torch.nonzero(output_role_indices, as_tuple=True)[0]
        output_filler_indices = output_filler_indices[output_role_indices != 0]
        output_role_indices = output_role_indices[output_role_indices != 0]
        target = torch.sparse_coo_tensor(indices=torch.stack((output_batch_indices, output_role_indices)),
                                         values=output_filler_indices, size=(bsz, self.tpr.d_role))

        if self.sparse:
            input_ = input_.coalesce()
            target = target.coalesce()
        else:
            input_ = input_.to_dense()
            target = target.to_dense()

        input_filler_root_embeddings = None
        input_filler_root_mask = None
        if (self.model.root_prediction_type == RootPredictionType.POSITION_ATTN_OVER_INPUTS or
                self.model.root_prediction_type == RootPredictionType.QK_ATTN_OVER_INPUTS):
            input_filler_root_indices = torch.zeros(bsz, self.max_input_length, dtype=torch.long)
            input_filler_root_indices[:, :len_] = batch['input_fillers'][:, :, 0]
            input_filler_root_mask = input_filler_root_indices == 0
            input_filler_root_indices = input_filler_root_indices.to(self.device, non_blocking=True)
            input_filler_root_mask = input_filler_root_mask.to(self.device, non_blocking=True)
            input_filler_root_embeddings = self.tpr.filler_emb(input_filler_root_indices)

        model = self.model if not use_ddp_module else self.model.module
        # TODO: model shouldn't print anything during debug, it should return the things that get printed out here
        #  once we do this we can also remove decoded_tpr_to_tree_fn from this funciton call
        output, debug_info, entropies = model(
            self.tpr(input_),
            bsz,
            debug=debug,
            calculate_entropy=False,
            custom_memory_set=self.use_custom_memory,
            vocab_info=self.vocab_info if debug else None,
            decoded_tpr_to_tree_fn=self.decoded_tpr_to_tree_fn if debug else None,
            input_filler_root_embeddings=input_filler_root_embeddings,
            input_filler_root_mask=input_filler_root_mask,
        )

        decoded = self.tpr.unbind(output, decode=True)
        fully_decoded = self.decoded_tpr_to_tree_fn(decoded)
        if self.sparse:
            # I suspect that this way of selecting the values in decoded which appear in target is non-optimal,
            # but I can't think of a better way off the top of my head.
            # This mask works by checking that both the batch index and role index for target and decoded are equal,
            # and then checks that this is true for any position in decoded.
            target_not_padding_mask = target.values() != self.pad_idx
            pairwise_mask = (target.indices()[:, target_not_padding_mask].T == fully_decoded.indices().T[:, None]).all(-1)
            target_mask = pairwise_mask.any(0)
            decoded_mask = pairwise_mask.any(-1)

            # decoded_mask returns the values in decoded that are at indices which exist in target. However, if target
            # has a value at an index which does not exist in decoded, we still want to include that value in the loss.
            # Wherever there is a value in target and not in decoded, decoded would have returned 0 since it is a
            # sparse tensor, so we initialize everything to zero and then fill in the values that exist in decoded.
            logits = torch.zeros(
                target.values()[target_not_padding_mask].shape[0],
                self.tpr.num_output_fillers,
                device=output.indices().device
            )
            logits[target_mask] = decoded.values()[decoded_mask]
            # We don't ever want to predict the padding token. Also, this prevents gradient from following through the
            # padding embedding which would make it non-zero.
            logits[:, self.output_indices_mask] = -float('inf')

            weights = None
            if self.cross_entropy_weighting == 'inverse':
                # Add 1 so that we don't divide by zero
                frequencies = torch.bincount(target.values(), minlength=len(self.index2vocab)) + 1
                weights = 1. / frequencies
            elif self.cross_entropy_weighting == 'balanced':
                # Add 1 so that we don't divide by zero
                frequencies = torch.bincount(target.values(), minlength=len(self.index2vocab)) + 1
                weights = 1. / frequencies
                weights = weights / weights.sum() * len(frequencies)
            elif self.cross_entropy_weighting == 'sqrt_inverse':
                # Add 1 so that we don't divide by zero
                frequencies = torch.bincount(target.values(), minlength=len(self.index2vocab)) + 1
                weights = 1. / torch.sqrt(frequencies)
            elif self.cross_entropy_weighting == 'sqrt_inverse_balanced':
                # Add 1 so that we don't divide by zero
                frequencies = torch.bincount(target.values(), minlength=len(self.index2vocab)) + 1
                weights = 1. / torch.sqrt(frequencies)
                weights = weights / weights.sum() * len(frequencies)

            per_token_loss = nn.functional.cross_entropy(
                logits, target.values()[target_not_padding_mask], weight=weights, reduction='none'
            )
            loss = per_token_loss.mean()

            # Calculate the token accuracy
            predicted_tokens = torch.zeros(target.values()[target_not_padding_mask].shape[0], dtype=torch.int64,
                                           device=fully_decoded.indices().device)
            predicted_tokens[target_mask] = fully_decoded.values()[decoded_mask]

            correct_tokens = predicted_tokens == target.values()[target_not_padding_mask]
            masked_correct_tokens = torch.sparse_coo_tensor(indices=target.indices()[:, target_not_padding_mask], values=correct_tokens,
                                                            size=(bsz, self.tpr.num_roles))
            correct_sequences = torch.sparse.sum(masked_correct_tokens, 1, dtype=torch.int).values() == torch.bincount(
                target.indices()[0, target_not_padding_mask], minlength=bsz)
            token_total = target.values()[target_not_padding_mask].numel()

            perplexity = torch.tensor(0., device=self.device)
            if self.nt_token_index:
                actual_tokens_mask = torch.logical_and(
                    target.values()[target_not_padding_mask] != self.nt_token_index,
                    target.values()[target_not_padding_mask] != self.eob_token_index
                )

                perplexity = torch.exp(per_token_loss[actual_tokens_mask].mean())
        else:
            logits = decoded
            # We don't ever want to predict the padding token. Also, this prevents gradient from following through the
            # padding embedding which would make it non-zero.
            logits[:, :, self.output_indices_mask] = -float('inf')
            empty_positions = target == self.pad_idx
            loss = self.xent_loss(logits[~empty_positions], target[~empty_positions])
            correct_tokens = torch.logical_and(fully_decoded == target, ~empty_positions)
            correct_sequences = torch.sum(correct_tokens, -1) == torch.sum(~empty_positions, -1)
            token_total = (~empty_positions).sum()
            # TODO: calculate perplexity for the non-sparse path if I need it
            perplexity = torch.tensor(0., device=self.device)

        return (loss, correct_tokens.sum(), token_total, correct_sequences, bsz, debug_info, entropies, output,
                perplexity)
