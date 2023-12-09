# Define the parameters in a dictionary
import os
import glob
import pandas as pd
from functools import partial


def run_training(out_name,
    kwargs,):
    # layerwise_pe, use_residual = args

    pe_type='original'
    params = {
        'max_iters': 5000, 
        'lr_decay_iters': 5000, # keep the original training schedule
        'general_seed': 888,
        'out_dir': 'outputs',
        'pe_type': pe_type,  # or 'sin'
        # 'learning_rate': 0.00055221,
        # 'warmup_iters': 422,

        # 'learning_rate': 0.00038441, # 1202
        # 'warmup_iters': 797,
        
        'learning_rate': 0.00026441, # 1202 manual
        'warmup_iters': 797,

        'use_residual': kwargs['use_residual'],
        # 'layerwise_pe': kwargs['layerwise_pe'],
        # 'permute': ,
        'not_causal': kwargs['not_causal'],

    }

    params['out_name'] = out_name

    
    # Construct the output directory and other variables
    output_directory = os.path.join(params['out_dir'], f"{params['out_name']}/addition_reverse_sd{params['general_seed']}")
    wandb_run_name = f"addition_reverse_sd{params['general_seed']}"
    wandb_project = params['out_name']

    # Construct the command using parameters from the dictionary
    # layerwise_pe_repr = '['+','.join(map(str,params['layerwise_pe']))+']' if type(params['layerwise_pe']) is list else str(params['layerwise_pe'])
    use_residual_repr = '['+','.join(map(str,params['use_residual']))+']' if type(params['use_residual']) is list else str(params['use_residual'])
    # permute_repr = '['+','.join(map(str,params['permute']))+']' if type(params['permute']) is list else str(params['permute'])
    not_casual_repr = '['+','.join(map(str,params['not_causal']))+']' if type(params['not_causal']) is list else str(params['not_causal'])
    command_params = {
        'use_pe': params['pe_type'],
        'use_residual': use_residual_repr,
        'max_iters': params['max_iters'],
        'lr_decay_iters': params['lr_decay_iters'],
        # 'layerwise_pe': layerwise_pe_repr,
        # 'permute': permute_repr,
        'not_causal': not_casual_repr,
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

    
    # ==================== 1203 ====================
    out_name = f"out4_1203"

    # use_residual_list = [[2,3,4,5], [2,3,4,5], [2,3,4,5]]
    # not_causal_list = [[1,],[0,], [0, 1]]

    # control

    use_residual_list = [[0,1,2,3,4,5]]
    not_causal_list = [[0,1,2,3,4,5]]
    # no SC[i] SC[i+1] yes lwp=True
    # use_residual_list = [[j for j in range(6) if j not in [i, i+1]] for i in range(5)]
    # layerwise_pe_list = [True for i in range(5)]
    
    os.makedirs(f"{out_dir}/{out_name}", exist_ok=True)

    # do a multi-processing, using 2 processes at a time
    from multiprocessing import Pool
    from functools import partial
    pool = Pool(1)
    func = partial(run_training, out_name)
    args = [{
        'not_causal': not_causal_list[i],
        'use_residual': use_residual_list[i],
        # 'layerwise_pe_list': layerwise_pe_list[i],
    } for i in range(len(not_causal_list))]
    pool.map(func, args)
    pool.close()