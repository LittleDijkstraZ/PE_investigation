# Define the parameters in a dictionary
import os
import glob
import pandas as pd
import wandb
from functools import partial


def run_training(out_name,
    args,
    ):
    layerwise_pe, use_residual = args
    pe_type='original'
    params = {
        'max_iters': 5000, 
        'lr_decay_iters': 5000, # keep the original training schedule
        'general_seed': 888,
        'out_dir': 'outputs',
        'pe_type': pe_type,  # or 'sin'
        'learning_rate': 0.00055221,
        'warmup_iters': 422,
        'use_residual': use_residual,
        'layerwise_pe': layerwise_pe,

    }

    params['out_name'] = out_name

    
    # Construct the output directory and other variables
    output_directory = os.path.join(params['out_dir'], f"{params['out_name']}/addition_reverse_sd{params['general_seed']}")
    wandb_run_name = f"addition_reverse_sd{params['general_seed']}"
    wandb_project = params['out_name']

    # Construct the command using parameters from the dictionary
    layerwise_pe_repr = '['+','.join(map(str,params['layerwise_pe']))+']' if type(params['layerwise_pe']) is list else str(params['layerwise_pe'])
    use_residual_repr = '['+','.join(map(str,params['use_residual']))+']' if type(params['use_residual']) is list else str(params['use_residual'])
    command_params = {
        'use_pe': params['pe_type'],
        'use_residual': use_residual_repr,
        'max_iters': params['max_iters'],
        'lr_decay_iters': params['lr_decay_iters'],
        'layerwise_pe': layerwise_pe_repr,
        'out_dir': output_directory,
        'wandb_run_name': wandb_run_name,
        'learning_rate': params['learning_rate'],
        'warmup_iters': params['warmup_iters'],
        'wandb_project': wandb_project
    }

    command = "python3 train.py pe_info/config2_pe/addition/reverse/jason_train_addition_bal.py "
    for key, value in command_params.items():
        command += f" --{key}={value}"
    print(command)
    os.system(command)




if __name__ == "__main__":
    out_dir = "./outputs"
    out_name = f"out4_1201"
    os.makedirs(f"{out_dir}/{out_name}", exist_ok=True)
    
    # # no SC[i] yes lwp[i]
    # use_residual_list = [[j for j in range(6) if j != i] for i in range(6)]
    # layerwise_pe_list = [[i] for i in range(6)]

    # control no SC[i] yes lwp[i]
    use_residual_list = [[j for j in range(6) if j != i] for i in range(6)]
    layerwise_pe_list = [False,]*6


    #
    # do a multi-processing, using 2 processes at a time
    from multiprocessing import Pool
    from functools import partial
    pool = Pool(1)
    func = partial(run_training, out_name)
    pool.map(func, list(zip(layerwise_pe_list, use_residual_list)))
    pool.close()