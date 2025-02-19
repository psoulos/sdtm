# datasets used for training in the Blackboard project:
    nc_pat: passive/active/logical sentence transformations (also includes car_cdr_cons, car_cdr_rcons diagnostic tasks)
    nc_tiny: small dataset that test systematicity
    nc_math: simple addition and subtraction 
    SCAN (https://arxiv.org/pdf/1711.00350v3.pdf)
    cogs  (https://tallinzen.net/media/papers/kim_linzen_2020_emnlp.pdf)

# Automatic download:
    - if you launch a local training run, it will download the dataset/task specified for the run, if the files are not found on your local machine
    - if you launch a training run thru XT, it will download the dataset/task specified for the run onto the assigned compute node
    - the location of the dataset files is defined by the --data_path command line argument (it defaults to $HOME/.data)
    - other command line arguments for data: --dataset_name, --dataset_version, --task, --data_filter, --data_filter2
    
    - if the environment variable HOME has not been set for your run, it defaults to:
            Window: the %USERPROFILE% directory 
            Linux: the ~/ directory 

# Explict download:
    - you can use the following command to explictly download a full dataset (all tasks):

        xt download $data/datasetname/version localfolder

    - for example, to download v15 of the nc_pat dataset, you can use the command:

        windows: > xt download $data/nc_pat/v15 d:/.data/nc_pat/v15
        linux:   $ xt download $$data/nc_pat/v15 ~/.data/nc_pat/v15
