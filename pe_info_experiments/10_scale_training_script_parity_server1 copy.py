# Define the parameters in a dictionary
import os
import glob
from click import command
import pandas as pd
from functools import partial


def run_training(out_name,
    kwargs,):
    # layerwise_pe, use_residual = args

    # pe_type='original'
    params = {
        'max_iters': kwargs['max_iters'] if 'max_iters' in kwargs else 5000, # 10000
        # 'max_iters': 10000, # increase to see if hard-to-converge cases can converge
        'lr_decay_iters':  kwargs['max_iters'] if 'max_iters' in kwargs else 5000, # keep the original training schedule
        'general_seed': kwargs['general_seed'] if 'general_seed' in kwargs else 888,
        'out_dir': 'outputs',
        'pe_type': kwargs['use_pe'],  # or 'sin'
        # 'learning_rate': 0.00055221,
        # 'warmup_iters': 422,

        # 'learning_rate': 0.00038441, # 1202
        # 'warmup_iters': 797,
        
        # 'learning_rate': 0.00026441, # 1202 manual
        # 'warmup_iters': 797,

        # 'learning_rate': 0.000125, # 0126 manual
        # 'warmup_iters': 600,
        
        # 'learning_rate': 0.001, # original setting
        # 'warmup_iters': 200,

        # 'learning_rate': 0.00026441, # 0126 try to preven rank degen
        # 'warmup_iters': 400,

        ### for non causal task
        # 'learning_rate': 0.000026441, # 0126 try to preven rank degen
        # 'warmup_iters': 200,
        'learning_rate': kwargs['learning_rate'] if 'learning_rate' in kwargs else 0.0002644, # 0126 try to preven rank degen
        'warmup_iters': kwargs['max_iters']*0.1 if 'max_iters' in kwargs else 400,


        'use_residual': kwargs['use_residual'],
        # 'layerwise_pe': kwargs['layerwise_pe'],
        # 'permute': ,
        'not_causal': kwargs['not_causal'],
    }

    params['out_name'] = out_name
    if 'message' in kwargs and len(kwargs['message']) > 0:
        params['message'] = '_'+kwargs['message'] 
    else:
        params['message'] = ''
    
    # Construct the output directory and other variables
    # wandb_run_name = f"addition_reverse_sd{params['general_seed']}{params['message']}"
    # wandb_run_name = f"parity_sd{params['general_seed']}{params['message']}"
    # wandb_run_name = f"sumd_sd{params['general_seed']}{params['message']}"\

    choice = kwargs['choice']
    wandb_run_name = f"{choice}_sd{params['general_seed']}{params['message']}"
        
        

    output_directory = os.path.join(params['out_dir'], f"{params['out_name']}/{wandb_run_name}")
    current_exist = os.listdir(os.path.join(params['out_dir'], f"{params['out_name']}"))
    tmp_pe = params['pe_type'] if params['pe_type'] != 'original' else '' 
    for dir in current_exist:
        if f"sd{params['general_seed']}" in dir \
            and f'_res={params["use_residual"]}' in dir\
            and f"{tmp_pe}" in dir:
            print(f"Already exist: {dir}")
            return
    print(f'need to run', f"sd{params['general_seed']}", f'_res={params["use_residual"]}', f"{tmp_pe}") 
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
        'general_seed': params['general_seed'],
        'not_causal': not_casual_repr,
        'out_dir': output_directory,
        'wandb_run_name': wandb_run_name,
        'learning_rate': params['learning_rate'],
        'warmup_iters': params['warmup_iters'],
        'wandb_project': wandb_project,
    }
    if 'use_flash' in kwargs:
        command_params['use_flash'] = kwargs['use_flash']
    if 'n_layer' in kwargs:
        command_params['n_layer'] = kwargs['n_layer']
    if 'no_att_residual' in kwargs:        
        command_params['no_att_residual'] = kwargs['no_att_residual']
    if 'no_mlp_residual' in kwargs:
        command_params['no_mlp_residual'] = kwargs['no_mlp_residual']
    if 'batch_size' in kwargs:
        command_params['batch_size'] = kwargs['batch_size']
    if 'causal_training' in kwargs:
        command_params['causal_training'] = kwargs['causal_training']
    if 'save_best_loss' in kwargs:
        command_params['save_best_loss'] = kwargs['save_best_loss']
    if 'autoregressive_training' in kwargs:
        command_params['autoregressive_training'] = kwargs['autoregressive_training']

    # command = "python3 train.py pe_info/config2_pe/addition/reverse/jason_train_addition_bal.py "
    # command = "python3 train.py pe_info/config2_pe/parity/jason_train_addition_bal.py "
    # command = "python3 train.py pe_info/config2_pe/sumd/jason_train_addition_bal.py "
    # command = "python3 train.py pe_info/config2_pe/oddc/jason_train_addition_bal.py "
    command = kwargs['command']
        

    for key, value in command_params.items():
        command += f" --{key}={value}"
    print(command)
    os.system(command)




if __name__ == "__main__":
    out_dir = "./outputs"

    
    # ==================== 1203 ====================
    # out_name = f"out3_control" # out4_1203 causal didn't converge.


    # current problems
    # 1. loss is weird


    # use_residual_list0 = [[0,3,4,5]]s
    # use_residual_list0 = [False]

    # not_causal_list = [[1,],[0,], [0, 1]]

    # control

    # use_residual_list = [[3, 4, 5]]
    use_residual_list1 = [[i for i in range(6) if i not in [j, j+1, j+2]] for j in range(4)]
    use_residual_list2 = [[i for i in range(6) if i not in [j, j+1]] for j in range(5)]

    # use_residual_list2 = [[i for i in range(6) if i not in [j, j+2]] for j in range(4)]
    # use_residual_list3 = [[i for i in range(6) if i not in [j,]] for j in range(6)]
    # use_residual_list4 = [[i for i in range(6)]]
    use_residual_list4 = [[3,4,5]]

    
    # use_residual_list4 = [True]
    use_residual_list_all = [[]] \
        + [[i for i in range(6) if i not in [j, j+1, j+2, j+3, j+4]] for j in range(2)] \
        + [[i for i in range(6) if i not in [j, j+1, j+2, j+3]] for j in range(3)] \
        + [[i for i in range(6) if i not in [j, j+1, j+2]] for j in range(4)] \
        + [[i for i in range(6) if i not in [j, j+1]] for j in range(5)] \
        + [[i for i in range(6) if i not in [j,]] for j in range(6)] \
        + [[i for i in range(6)]]
    # use_residual_list3 = [[i for i in range(6) if i not in [j,]] for j in range(2, 6)]
    seeds = [240+i for i in range(0, 5)]

    commands_dict = {
        # "add3": "python3 train.py pe_info/config2_pe/addition/reverse/jason_train_addition_bal.py ",
        "add3_nc": "python3 train.py pe_info/config2_pe/addition/reverse/jason_train_addition_bal.py ",
        "mod3" : "python3 train.py pe_info/config2_pe/mod3/jason_train_addition_bal.py ",
        "mod3_nc" : "python3 train.py pe_info/config2_pe/mod3/jason_train_addition_bal.py ",

        "modp" : "python3 train.py pe_info/config2_pe/modp/jason_train_addition_bal.py ",
        "modp_nc" : "python3 train.py pe_info/config2_pe/modp/jason_train_addition_bal.py ",

        "parity": "python3 train.py pe_info/config2_pe/parity/jason_train_addition_bal.py ",
        "parity_nc_repeat": "python3 train.py pe_info/config2_pe/parity/jason_train_addition_bal.py ",
        "paridy": "python3 train.py pe_info/config2_pe/paridy/jason_train_addition_bal.py ",
        "paridy_nc": "python3 train.py pe_info/config2_pe/paridy/jason_train_addition_bal.py ",

        "sumd_c": "python3 train.py pe_info/config2_pe/sumd/jason_train_addition_bal.py ",
        "oddc": "python3 train.py pe_info/config2_pe/oddc/jason_train_addition_bal.py "
    }
    for seed in seeds:

        for choice in ["paridy", "paridy_nc"]:
            # choice = "mod3_nc"
            causal_training = False
            autoregressive_training = False

            # batch_size = 2048 
            # max_iters = 2000 
            # learning_rate = 0.000026441 
            # warmup_iters = 400 

            batch_size = 2048 
            max_iters = 4000 
            learning_rate = 0.000026441 
            warmup_iters = 400 


            # batch_size = 2048 if not causal_training else 256
            # max_iters = 2000 if not causal_training else 5000
            # learning_rate = 0.000026441 if not causal_training else  0.00026441
            # warmup_iters = 200 if not causal_training else 400
            # batch_size = 256
            # max_iters = 5000 
            # learning_rate = 0.0001
            # warmup_iters = 400

            
            # for use_pe in ['nope', 'original']: # 'original''nope', 

            # for use_residual_list in [use_residual_list2, use_residual_list3]: # use_residual_list1, use_residual_list2, 
            # for use_residual_list in [use_residual_list1, use_residual_list2]: # use_residual_list1, use_residual_list2, 
            for use_residual_list in [use_residual_list_all]: # use_residual_list1, use_residual_list2, 
                

                # no_att_residual_list = [True]
                # no_mlp_residual_list = [True]
                bval = True if 'nc' in choice else False
                not_causal_list = [bval] * len(use_residual_list)
                # not_causal_list = [False]
                # bs = [512]

                # n_layers = [,]
                # no SC[i] SC[i+1] yes lwp=True
                # use_residual_list = [[j for j in range(6) if j not in [i, i+1]] for i in range(5)]
                # layerwise_pe_list = [True for i in range(5)]
                

                # do a multi-processing, using 2 processes at a time
                from multiprocessing import Pool
                from functools import partial
                

                # for seed in [222, 333, 444]:
                # for use_pe in ['nope', 'original']: # 'original''nope',
                # for use_pe in ['original', 'nope']: # 'original''nope',
                # for use_pe in ['nope']: # 'original''nope',
                for use_pe in ['original', 'nope']: # 'original''nope',


                    out_name = f"{choice}_nope_residual_exp" if use_pe=='nope' else f"{choice}_residual_exp" # out4_1203 causal didn't converge.
                    os.makedirs(f"{out_dir}/{out_name}", exist_ok=True)

                    pool = Pool(1)
                    func = partial(run_training, out_name)
                    args = [{
                        'not_causal': not_causal_list[i],
                        'use_residual': use_residual_list[i],
                        # 'n_layer': n_layers[i],
                        'use_pe': use_pe,
                        'general_seed': seed,
                        'choice': choice,

                        'causal_training': causal_training,
                        'batch_size': batch_size, 
                        'max_iters': max_iters,
                        'learning_rate': learning_rate,
                        'warmup_iters': warmup_iters,
                        
                        'command': commands_dict[choice],  
                        'save_best_loss': True,
                        'autoregressive_training': autoregressive_training,

                        # 'no_att_residual': no_att_residual_list[i],
                        # 'no_mlp_residual': no_mlp_residual_list[i],
                        # 'batch_size': bs[i],
                        # 'message': '',
                        # 'layerwise_pe': True,
                        # 'use_flesh': True,
                        # 'layerwise_pe_list': layerwise_pe_list[i],
                    } for i in range(len(use_residual_list))]
                    # for arg in args:
                        # func(arg)
                    pool.map(func, args)
                    pool.close()   

                # implement a teacher forcing