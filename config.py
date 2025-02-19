import argparse
import random

from models import RootPredictionType
from node import AddEobTokens


def parse_args(args_list=None):
    parser = argparse.ArgumentParser()

    _define_model_args(parser)
    _define_data_args(parser)
    _define_train_args(parser)
    _define_optim_args(parser)
    _define_io_args(parser)
    _define_system_args(parser)

    args = parser.parse_args(args_list)

    if args.steps is None and args.epoch is None:
        assert False, 'Either --steps or --epoch must be set.'
    elif args.steps is not None and args.epoch is not None:
        assert False, 'Only one of --steps or --epoch can be set.'

    assert not (args.filler_threshold and args.max_filled_roles), ('Cannot set both --filler_threshold and '
                                                                   '--max_filled_roles.')

    if args.router_hidden_dim is None:
        args.router_hidden_dim = args.ctrl_hidden_dim * 4

    if args.filler_map_location == 'None':
        args.filler_map_location = None

    if args.filler_dropout_location == 'None':
        args.filler_dropout_location = None

    if args.filler_noise_location == 'None':
        args.filler_noise_location = None

    if args.seed is None:
        # get a specific seed that we can report with the run
        args.seed = random.randint(0, 65535)

    args.d_role = 2 ** args.max_tree_depth if args.d_role is None else args.d_role
    if args.use_vocab_info:
        args.add_eob_tokens = AddEobTokens.UNARY
    else:
        args.add_eob_tokens = AddEobTokens.ALL

    return args


def _define_model_args(parser):
    parser.add_argument('--d_filler', type=int, default=None,
                        help='dimension of filler vectors. Set to len(train_vocab) if None.')
    parser.add_argument('--d_role', type=int, default=None,
                        help='dimension of role vectors. Set to 2**max_tree_depth if None.')
    parser.add_argument('--dtm_layers', type=int, default=3,
                        help='Number of layers to operate on the DTM.')
    parser.add_argument('--op_dist_fn', type=str, default='softmax', choices=['softmax', 'gumbel'],
                        help='The operation distribution function.')
    parser.add_argument('--arg_dist_fn', type=str, default='softmax', choices=['softmax', 'gumbel'],
                        help='The argument distribution function.')
    parser.add_argument('--sparse', type=int, default=1, help='Whether the roles are sparse')
    parser.add_argument('--filler_threshold', type=float, default=None,
                        help='The threshold for the fillers vectors in resulting TPRs. If None, no thresholding is'
                             ' done. This is used to prevent a sparse TPR from becoming dense through CONS writing very'
                             ' small fillers to every role over many steps. Should be something like 1e-2 or less but'
                             ' this is just a guess. Higher values will save more memory but may make training'
                             ' difficult.')
    parser.add_argument('--max_filled_roles', type=int, help='The maximum number of roles that can be filled. Any '
                                                             'roles over this amount will be dropped. Currently roles '
                                                             'with the largest filler magnitude are kept.')
    parser.add_argument('--custom_memory', type=int, default=1, help='Whether to use our custom memory implementation '
                                                                     'which pre-allocates the memory to avoid '
                                                                     'torch.cat().')
    parser.add_argument('--transformer_norm_first', type=int, default=1)
    parser.add_argument('--transformer_activation', type=str, default='gelu')
    parser.add_argument('--transformer_nheads', type=int, default=4)
    parser.add_argument('--filler_emb_gain', type=float, default=1.0)
    parser.add_argument('--input_norm', type=str, default=None,
                        help='Whether to normalize the input. Options are [None, tpr_norm, ctrl_norm]')
    parser.add_argument('--router_dropout', type=float, default=0.0,
                        help='Router dropout')
    parser.add_argument('--router_hidden_dim', type=int, default=None,
                        help='Router hidden dim')
    parser.add_argument('--agent_layers_per_step', type=int, default=1,
                        help='The number of encoding layers for each step'
                             ' of the dtm')
    parser.add_argument(
        '--root_prediction_type',
        type=RootPredictionType,
        default=RootPredictionType.LINEAR,
        choices=list(RootPredictionType),
        help='See @models.RootPredictionType for options. This is the method used to predict the root filler.'
    )
    parser.add_argument('--proj_filler_to_unit_ball', action='store_true',
                        help='Whether to ensure that each filler vector has L2 norm 1')
    parser.add_argument('--learn_filler_embed', type=int, default=1,
                        help='Whether to learn filler embeddings. If 0, the filler embeddings are fixed to the'
                             ' orthogonal.')
    parser.add_argument('--learn_empty_filler', action='store_true',
                        help='Whether the empty filler should be learned like any other filler. Otherwise it is fixed.'
                             'Also see --empty_filler_initialization.')
    parser.add_argument('--empty_filler_initialization', default='zero', choices=['zero', 'random'])
    parser.add_argument('--ctrl_type', type=str, default='set_transformer',
                        help='The transformation to use for the control state. Options are [linear, conv, conv_mlp,'
                             ' set_transformer]')
    parser.add_argument('--n_conv_kernels', type=int, default=4,
                        help='The number of output kernels to use for ctrl_type conv and conv_mlp.')
    parser.add_argument('--ctrl_hidden_dim', type=int, default=64,
                        help='Ctrl hidden dim')
    parser.add_argument('--include_empty_tpr', type=int, default=0,
                        help='Include the empty TPR in memory. If 0, the empty TPR is not in memory.')
    parser.add_argument('--predefined_operations_are_random', action='store_true',
                        help='Whether the car/cdr/cons matrices are calculated exactly or learnable random matrices')
    parser.add_argument('--filler_map_location', default=None, choices=[None, 'pre_dtm', 'post_dtm', 'operation',
                                                                        'pre_shrink', 'None'])
    parser.add_argument('--filler_map_type', default=None, choices=[None, 'linear', 'mlp'])
    parser.add_argument('--hardcode_cons_root_token', type=str,
                        help='If set, the root filler of the cons operation will always be this token\'s embedding.'
                             '\'-1\' is a special value which means the empty filler.')
    parser.add_argument(
        '--cons_only',
        type=int,
        default=0,
        help='Only use the cons operation. This disables the car and cdr operations.'
    )
    parser.add_argument(
        '--add_eob_to_memory',
        type=int,
        default=0,
        help='Add the <EOB> token to the starting memory'
    )
    parser.add_argument(
        '--num_extra_tokens_in_memory',
        type=int,
        default=0,
        help=('The number of extra tokens to add to the memory. These tokens can be blended together to produce '
              'arbitrary embeddings')
    )
    parser.add_argument(
        '--tied_io_languages',
        type=int,
        default=1,
        help='Whether the vocabularies and embeddings should be shared between the input and output languages.'
    )
    parser.add_argument('--filler_dropout', type=float, default=0.0, help='Filler dropout')
    parser.add_argument(
        '--filler_dropout_location',
        type=str,
        default=None,
        choices=[None, 'None', 'input', 'pre_output'],
        help='Where to apply filler dropout.'
    )
    parser.add_argument(
        '--new_tree_filler_dropout1d',
        type=float,
        default=0.0,
        help='Dropout entire fillers for the output of the interpreter.'
    )

    # TODO: do we also want to experiment with filler noise injection pre_output?
    parser.add_argument(
        '--filler_noise_location',
        type=str,
        default=None,
        choices=[None, 'None', 'input',],
        help='Where to apply filler noise.'
    )
    parser.add_argument('--filler_noise_std', type=float, default=0.0, help='Filler noise std')
    parser.add_argument('--arg_noise_std', type=float, default=0.0, help='Argument noise std')
    parser.add_argument(
        '--is_agent_universal',
        type=int,
        default=0,
        help='Whether the agent uses the same parameters for each step.'
    )
    parser.add_argument(
        '--positional_embedding_type',
        type=str,
        default='sinusoidal',
        choices=['learned', 'sinusoidal'],
    )
    parser.add_argument(
        '--random_positional_max_len',
        type=int,
        default=0,
        help='The max length. This number must be equal to or greater than the longest sequence in the dataset. If '
             'set to 0, then random positional embeddings are not used. This hyperparameter is called "M"'
             ' in https://arxiv.org/pdf/2305.16843.pdf'
    )

def _define_data_args(parser):
    parser.add_argument('--data_dir', type=str, default='data/active_logical',
                        help='path from dataroot to task files')
    parser.add_argument(
        '--use_vocab_info',
        type=int,
        default=0,
        help='Whether to use information from the vocabulary to determine which symbols are terminal, '
             'unary branching, or binary branching. Otherwise we use EOB tokens added to the input and output trees.'
    )
    parser.add_argument('--data_filter', type=str, default=None, help='Sequences to filter for in the dataset')
    parser.add_argument('--max_train_examples', type=int, default=None, help='The number of training examples')
    parser.add_argument(
        '--max_tree_depth',
        type=int,
        default=6,
        help='max depth of input trees'
    )
    parser.add_argument('--output_lowercase', type=int, default=0, help='For SCAN only. This flag will align the '
                                                                        'input and output vocabulary for SCAN.')


def _define_train_args(parser):
    parser.add_argument('--validate_every_num_epochs', type=int, default=1,
                        help='How many training epochs do we wait before validating. The default is to validate after'
                             'every training epoch. This is useful when max_train_examples is set to a small number'
                             'so that we don\'t spend too much time validating.')
    parser.add_argument('--tpr_loss_type', default='filler_xent', choices=['filler_xent', 'tpr_mse'])
    parser.add_argument('--epoch', type=int, default=None,
                        help='Number of training epochs. Either this or --steps must be set.')
    parser.add_argument('--steps', type=float, default=None,
                        help='Number of training steps. Either this or --epoch must be set.')
    parser.add_argument('--batch_size', type=int, default=64, )
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument('--early_stop_epochs', type=int, default=10,
                        help='The number of epochs we will wait for a new best validation loss before stopping'
                             ' training')
    parser.add_argument(
        '--cross_entropy_weighting',
        type=str,
        default=None,
        choices=[None, 'inverse', 'balanced', 'sqrt_inverse', 'sqrt_inverse_balanced'],
    )
    parser.add_argument(
        '--entropy_regularization_coefficient',
        type=float,
        default=0.0,
        help='The coefficient for the entropy regularization term in the loss.'
    )


def _define_optim_args(parser):
    parser.add_argument('--num_warmup_steps', type=float, default=0,
                        help='Number of warmup steps to linearly increase the learning rate from 0 to lr.')
    parser.add_argument('--gclip', type=float, default=1.)
    parser.add_argument('--optimizer', default='adam', choices=['sgd', 'adam', 'rmsprop', 'lamb'])
    parser.add_argument('--optim_beta1', type=float, default=0.9)
    parser.add_argument('--optim_beta2', type=float, default=0.95)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--scheduler', default='none', choices=['cosine', 'exponential', 'none'])
    parser.add_argument('--scheduler_gamma', type=float, default=1,
                        help='The gamma to apply if the scheduler is exponential.')
    parser.add_argument('--wd', type=float, default=1e-2)


def _define_io_args(parser):
    parser.add_argument('--wandb_name', type=str, default=None)
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--wandb_group', type=str, default=None)
    parser.add_argument('--train_log_freq', type=float, default=-1.,
                        help='training log frequency in steps, -1 logs at the end of each epoch')
    parser.add_argument('--out_dir', type=str, default='out',
                        help='The output directory. WARNING: changing this from the default will cause the model '
                             'checkpoint to not be mirrored by xt.')
    parser.add_argument('--debug', type=int, default=1, help='1 [debug], 0 [off].')
    parser.add_argument('--log_grad_norm', action='store_true',
                        help='Whether to log gradient norms during training.')
    parser.add_argument('--save_file', type=str, default='result.tsv')
    parser.add_argument('--best_checkpoint_file', type=str, default='best_checkpoint.pt')
    parser.add_argument('--most_recent_checkpoint_file', type=str, default='most_recent_checkpoint.pt')
    parser.add_argument('--resume', action='store_true', help='Whether to resume training from a checkpoint.')
    parser.add_argument('--log_level', type=str, default='INFO', choices=['VERBOSE', 'DEBUG', 'INFO', 'WARNING',])
    parser.add_argument(
        '--test_most_recent_checkpoint',
        type=int,
        default=0,
        help='Whether to run the final test results on the best checkpoint as measured by validation loss of the most'
             ' recent checkpoint'
    )

def _define_system_args(parser):
    parser.add_argument('--num_workers', type=int, default=4, help='The number of workers to load data')
    parser.add_argument('--fp16', action='store_true')
