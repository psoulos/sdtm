## Create the environment and install the packages
`conda create -n sdtm-public-in-progress python=3.8`

Install pytorch for whatever your system is via https://pytorch.org/get-started/locally/

`pip install -r requirements.txt`

## Test command to run sDTM on active<->logical
`python main.py --dtm_layers=16 --data_dir=data/active_logical --batch_size=16 --train_log_freq=20 --max_tree_depth=13 --sparse=1 --learn_filler_embed=0 --use_vocab_info=1 --tied_io_languages=1 --max_filled_roles=1028 --ctrl_hidden_dim=64 --lr=1e-4 --optim_beta2=0.95 --optim_beta1=0.9 --gclip=1 --wd=1e-1 --transformer_nheads=4 --router_dropout=0.1 --num_warmup_steps=10000 --scheduler=cosine --steps=20000`

## Test command to run sDTM on SCAN
`python main.py --data_dir=data/SCAN/simple_split/separate_to_parsed --test_most_recent_checkpoint=1 --max_tree_depth=7 --max_filled_roles=256 --dtm_layers=14 --hardcode_cons_root_token="<NT>" --validate_every_num_epochs=3 --use_vocab_info=1 --random_positional_max_len=18 --tied_io_languages=1 --add_eob_to_memory=1 --cons_only=1 --early_stop_epochs=1000 --filler_noise_location=input --filler_noise_std=1 --positional_embedding_type=sinusoidal --d_filler=64 --ctrl_hidden_dim=64 --lr=1e-4 --optim_beta2=0.95 --optim_beta1=0.9 --gclip=1 --wd=1e-1 --batch_size=128 --transformer_nheads=4 --router_dropout=0.1 --num_warmup_steps=10000 --scheduler=cosine --steps=50000 --train_log_freq=20`


## wandb
You can get $WANDB_API_KEY and $WANDB_USERNAME environment variables to use wandb. See the argument options `--use_wandb`, `--wandb_name`, and `--wandb_group`.
