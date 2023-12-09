# Define the parameters in a dictionary
import os
import glob
import pandas as pd
from functools import partial


def run_training(out_name, params_dict_updates):
    pe_type='original'
    wandb_run_name = f"addition_reverse"

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

        # 'use_residual': use_residual,
        # 'layerwise_pe': layerwise_pe,

    }
    for k in params_dict_updates:
        params[k] = params_dict_updates[k]

    params['out_name'] = out_name

    
    # Construct the output directory and other variables
    output_directory = os.path.join(params['out_dir'], f"{params['out_name']}/{wandb_run_name}")
    wandb_project = params['out_name']

    # Construct the command using parameters from the dictionary
    command_params = {
        'use_pe': params['pe_type'],
        'max_iters': params['max_iters'],
        'lr_decay_iters': params['lr_decay_iters'],
        'out_dir': output_directory,
        'wandb_run_name': wandb_run_name,
        'learning_rate': params['learning_rate'],
        'warmup_iters': params['warmup_iters'],
        'wandb_project': wandb_project
    }
    param_keys = ['use_residual', 'layerwise_pe', 'not_causal', 'general_seed']
    for k in param_keys:
        if k in params:
            command_params[k] = '['+','.join(map(str,params[k]))+']' if type(params[k]) is list else str(params[k])



    command = "python3 train.py pe_info/config2_pe/addition/reverse/jason_train_addition_bal.py "
    for key, value in command_params.items():
        command += f" --{key}={value}"
    print(command)
    os.system(command)




if __name__ == "__main__":
    out_dir = "./outputs"    
    # ==================== 1203 ====================
    out_name = f"out3_control"


    use_residual_list = [[0,1,2,3,4,5]]
    # not_causal_list = [[0,1,2,3,4,5]]
    # no SC[i] SC[i+1] yes lwp=True
    # use_residual_list = [[j for j in range(6) if j not in [i, i+1]] for i in range(5)]
    # layerwise_pe_list = [True for i in range(5)]

    args = [{
        # 'not_causal': not_causal_list[i],
        'use_residual': use_residual_list[i],
        # 'layerwise_pe_list': layerwise_pe_list[i],
    } for i in range(len(use_residual_list))]
    
    os.makedirs(f"{out_dir}/{out_name}", exist_ok=True)

    # do a multi-processing, using 2 processes at a time
    from multiprocessing import Pool
    from functools import partial
    pool = Pool(1)
    func = partial(run_training, out_name)

    pool.map(func, args)
    pool.close()