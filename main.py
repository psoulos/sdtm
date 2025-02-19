import logging
import os
import random
from argparse import Namespace
from typing import Optional

import torch.optim
import wandb
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

import data
from TPR_utils import TPR, decoded_tpr_to_tree_fn
from config import parse_args
from models import *
from trainer import Trainer


def setup_device() -> (bool, str, int, int):
    """Sets up the device depending on whether we are in DDP mode or not"""
    print(f'Cuda device count: {torch.cuda.device_count()}')
    print(f'Environment RANK: {os.environ.get("RANK", -1)}')
    is_ddp = int(os.environ.get('RANK', -1)) > 1  # is this a ddp run?
    if is_ddp:
        print('DDP mode')
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        ddp_rank = int(os.environ['RANK'])
        # ddp_world_size = int(os.environ['WORLD_SIZE'])
        init_process_group(backend='nccl')
        # TODO: I'm still not positive on how DDP affects the effective learning rate
        # args.lr *= ddp_world_size
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        ddp_local_rank = 0
        ddp_rank = 0
    return is_ddp, device, ddp_rank, ddp_local_rank


def setup_logging(args):
    numeric_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % args.log)
    logging.basicConfig()
    logger_ = logging.getLogger(__name__)
    logger_.setLevel(numeric_level)
    logger_.info(f'Log level set to {args.log_level}')
    # torch.cuda.set_sync_debug_mode(1)
    return logger_


def setup_wandb(args):
    if args.use_wandb and not os.getenv('WANDB_API_KEY'):
        raise ValueError('WANDB_API_KEY environment variable must be set to use wandb')
    if args.use_wandb and not os.getenv('WANDB_USERNAME'):
        raise ValueError('WANDB_USERNAME environment variable must be set to use wandb')

    wandb_name = args.wandb_name if args.wandb_name else ''
    wandb.init(
        project='DTM',
        entity=os.environ.get('WANDB_USERNAME'),
        config=args.__dict__,
        mode='online' if args.use_wandb else 'disabled',
        name=wandb_name,
        group=args.wandb_group,
        resume='allow',
    )


def setup_optimizer_and_scheduler(dtm: DiffTreeMachine, args: Namespace) -> \
        (torch.optim.Optimizer, Optional[torch.optim.lr_scheduler.LRScheduler]):
    trainable_params = list(filter(lambda p: p.requires_grad, dtm.parameters()))
    print('Trainable params: {}'.format(sum(p.numel() for p in trainable_params)))

    if args.optimizer == 'adam':
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=args.lr,
            weight_decay=args.wd,
            betas=(args.optim_beta1, args.optim_beta2),
        )
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(trainable_params, lr=args.lr, weight_decay=args.wd, momentum=args.optim_beta1)
    elif args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(trainable_params, lr=args.lr, weight_decay=args.wd)
    elif args.optimizer == 'lamb':
        import torch_optimizer
        optimizer = torch_optimizer.Lamb(
            trainable_params, lr=args.lr,
            weight_decay=args.wd,
            betas=(args.optim_beta1, args.optim_beta2)
        )
    else:
        raise ValueError(f'Unknown optimizer {args.optimizer}')

    if args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, verbose=False)
    elif args.scheduler == 'exponential':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, args.scheduler_gamma)
    else:
        scheduler = None

    return optimizer, scheduler


def convert_args_to_config(
    args,
    input_lang,
    output_lang,
    tpr,
    hardcode_cons_root_index,
    max_input_length
):
    """
    This function is a hack to fill in args with additional information that is expected by @DifferentiableTreeMachine
     and @NeuralTreeAgent
    """
    args.input_lang = input_lang
    args.output_lang = output_lang
    args.tpr = tpr
    args.hardcode_cons_root_index = hardcode_cons_root_index
    args.max_input_length = max_input_length
    args.d_model = args.ctrl_hidden_dim
    args.nhead = args.transformer_nheads
    args.dim_feedforward = args.router_hidden_dim
    args.dropout = args.router_dropout
    args.activation = args.transformer_activation
    args.layer_norm_eps = 1e-5
    args.pad_idx = 0


def main():
    args = parse_args()
    print(sorted(vars(args).items()))
    if args.fp16:
        raise NotImplementedError('fp16 is not implemented yet')
    is_ddp, device, global_rank, local_rank = setup_device()
    seed_offset = global_rank  # each process gets a different seed
    main_process = global_rank == 0  # this process will do logging, checkpointing etc.

    if main_process:
        logger = setup_logging(args)
        logger.debug(f'The main process is using device {device}')

    best_checkpoint = None
    most_recent_checkpoint = None

    if main_process:
        setup_wandb(args)
        os.makedirs(args.out_dir, exist_ok=True)

    if args.sparse:
        print('Sparse mode is on')

    random.seed(args.seed + seed_offset)
    torch.manual_seed(args.seed + seed_offset)

    data_dir = args.data_dir

    data_loaders, input_lang, output_lang = data.prepare_data_loaders(
        data_dir,
        args.max_tree_depth,
        args.add_eob_tokens,
        is_ddp,
        args.batch_size,
        args.num_workers,
        data_filter=args.data_filter,
        max_train_examples=args.max_train_examples,
        output_lowercase=args.output_lowercase,
        add_eob_to_memory=args.add_eob_to_memory,
        num_extra_tokens_in_memory=args.num_extra_tokens_in_memory,
    )

    input_vocab = set(input_lang.ind2vocab.values())  # Set of all unique tokens in the input vocabulary
    output_vocab = set(output_lang.ind2vocab.values())  # Set of all unique tokens in the output vocabulary

    input_unique_vocab = input_vocab - output_vocab

    # Merge the two languages
    for i, v in output_lang.ind2vocab.items():
        input_lang.add_word(v)
    output_lang = input_lang
    for data_loader in data_loaders.values():
        if data_loader:
            data_loader.dataset.output_lang = output_lang

    # We always want to mask 0 which is the pad index.
    # TODO: pad index should be a variable, I set it to 0 in a few places
    output_indices_mask = [0]
    # If the languages are not tied, anything unique to the input vocab should be masked out
    if not args.tied_io_languages:
        for i, v in output_lang.ind2vocab.items():
            if v in input_unique_vocab:
                output_indices_mask.append(i)

    print(f'Input language size: {len(input_lang.ind2vocab)}')
    print(f'Output language size: {len(output_lang.ind2vocab)}')

    max_input_length = -1
    for name, loader in data_loaders.items():
        if loader:
            max_input_length = max(max_input_length, loader.dataset.max_input_length)

    if not args.d_filler:
        args.d_filler = len(input_lang.ind2vocab)

    if args.steps is not None:
        args.epoch = math.ceil(args.steps / len(data_loaders['train']))
        print(f'Steps set to {args.steps} which is at most {args.epoch} epochs.')

    # Vocab info is used to determine when to stop decoding the tree. You can specify the terminal vocabulary,
    # or if you don't use vocab info, an <EOB> [end of branch] token will be added as the leaves of the dataset and used
    # as the terminal symbol.
    if args.use_vocab_info:
        vocab_info = data.get_vocab_info(args.data_dir, output_lang.ind2vocab.values(), )
    else:
        vocab_info = {
            'unary': (),
            'binary': (),
            'terminal': ('<EOB>',)
        }

    # TODO: tpr isn't used anywhere in this file, it should be initialized in DiffTreeMachine
    # TODO: the arguments to TPR and DiffTreeMachine are a mess, we should just pass args instead
    tpr = TPR(
        args,
        num_input_fillers=len(input_lang.ind2vocab),
        num_output_fillers=len(output_lang.ind2vocab),
        num_roles=2 ** args.max_tree_depth,
        d_filler=args.d_filler,
        d_role=args.d_role,
        filler_emb_gain=args.filler_emb_gain,
        learn_empty_filler=args.learn_empty_filler,
        tied_io_languages=args.tied_io_languages,
        empty_filler_initialization=args.empty_filler_initialization,
        device=device,
        sparse=args.sparse,
        nt_token_index=output_lang.vocab2ind.get('<NT>', None),
    ).to(device=device)

    hardcode_cons_root_index = None
    if args.hardcode_cons_root_token:
        if args.hardcode_cons_root_token == '-1':
            hardcode_cons_root_index = -1
        else:
            vocab2index = output_lang.vocab2ind
            assert args.hardcode_cons_root_token in vocab2index, (
                f'The token {args.harcode_cons_root_token} is not in the vocab.')
            hardcode_cons_root_index = vocab2index[args.hardcode_cons_root_token]
        logger.info(
            f'Hardcoding the root token to {args.hardcode_cons_root_token} with index {hardcode_cons_root_index}'
        )

    convert_args_to_config(args, input_lang, output_lang, tpr, hardcode_cons_root_index, max_input_length)

    dtm = DiffTreeMachine(args).to(device=device)

    optimizer, scheduler = setup_optimizer_and_scheduler(dtm, args)

    # TODO: make this a commandline arg
    watch_gradients = False
    if watch_gradients:
        wandb.watch(dtm, log='gradients', log_freq=1)

    # Compiling doesn't work yet
    compile_ = False
    if compile_:
        print("Compiling the model")
        dtm = torch.compile(dtm)

    if is_ddp:
        dtm = DDP(dtm, device_ids=[local_rank],)

    # TODO: this style is so ugly, we should just pass args to Trainer
    trainer = Trainer(
        dtm,
        tpr,
        data_loaders,
        optimizer,
        args.epoch,
        args.steps,
        args.num_warmup_steps,
        main_process,
        is_ddp,
        decoded_tpr_to_tree_fn(args.tpr_loss_type, sparse=args.sparse, output_indices_mask=output_indices_mask),
        torch.nn.CrossEntropyLoss(),
        device,
        output_lang.ind2vocab,
        vocab_info,
        args.use_wandb,
        args.validate_every_num_epochs,
        args.train_log_freq,
        early_stop_epochs=args.early_stop_epochs,
        pad_idx=0,
        sparse=args.sparse,
        scheduler=scheduler,
        gclip=args.gclip,
        lr=args.lr,
        out_dir=args.out_dir,
        best_checkpoint_file=args.best_checkpoint_file,
        most_recent_checkpoint_file=args.most_recent_checkpoint_file,
        use_custom_memory=args.custom_memory,
        cross_entropy_weighting=args.cross_entropy_weighting,
        entropy_regularization_coefficient=args.entropy_regularization_coefficient,
        max_input_length=args.max_input_length,
        nt_token_index=output_lang.vocab2ind.get('<NT>', None),
        eob_token_index=output_lang.vocab2ind.get('<EOB>', None),
        output_indices_mask=output_indices_mask,
    )

    if most_recent_checkpoint:
        logger.info('Resuming model, optimizer, and trainer from checkpoint')
        trainer.global_step = most_recent_checkpoint['step']
        trainer.epoch = most_recent_checkpoint['epoch']
        trainer.best_val_loss = most_recent_checkpoint['best_valid_loss']
        dtm.load_state_dict(most_recent_checkpoint['model'])
        optimizer.load_state_dict(most_recent_checkpoint['optimizer'])
        logger.info(f'Best valid loss from previous checkpoint: {trainer.best_val_loss}')

    trainer.train()
    if is_ddp:
        map_location = {'cuda:%d' % 0: 'cuda:%d' % global_rank}
    else:
        map_location = device

    if args.test_most_recent_checkpoint:
        logger.info('Testing the most recent checkpoint')
        dtm.load_state_dict(
            torch.load(os.path.join(args.out_dir, args.most_recent_checkpoint_file), map_location=map_location)['model']
        )
    else:
        logger.info('Testing the best checkpoint')
        dtm.load_state_dict(
            torch.load(os.path.join(args.out_dir, args.best_checkpoint_file), map_location=map_location)['model']
        )
    if main_process:
        trainer.test(data_loaders['valid'], print_incorrect_filename=os.path.join(args.out_dir, 'incorrect_valid.txt'))

        for test_set in ['test', 'eval_long', 'eval_new', 'eval_illformed']:
            if data_loaders[test_set]:
                loss, partial_acc, full_acc, _, perplexity = trainer.test(
                    data_loaders[test_set],
                    print_incorrect_filename=os.path.join(args.out_dir, f'incorrect_{test_set}.txt')
                )
                print(f'{test_set}\t  full_acc: {full_acc:.2f}, partial_acc: {partial_acc:.2f}, loss: {loss:.5f}')
                wandb.log(
                    {
                        f'final_{test_set}_loss': loss,
                        f'final_{test_set}_partial_acc': partial_acc,
                        f'final_{test_set}_full_acc': full_acc,
                        f'final_{test_set}_perplexity': perplexity,
                    }, step=trainer.global_step
                )
    if is_ddp:
        destroy_process_group()


if __name__ == '__main__':
    main()
