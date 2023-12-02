# Define the parameters in a dictionary
import os
import glob
import pandas as pd
import wandb
import optuna
from optuna.storages import JournalStorage, JournalFileStorage
from functools import partial


def run_tuning(trial, out_name):
    learning_rate = trial.suggest_float("lr", 1e-5, 6e-4, log=True)
    warmup_iters = trial.suggest_int("warmup_iters", 100, 1000, log=True)
    layerwise_pe = '[4]'
    params = {
        'max_iters': 2000, # train for only a small while, e.g., 8 steps(2000 iters)
        'lr_decay_iters': 5000, # keep the original training schedule
        'general_seed': 888,
        'out_dir': 'outputs',
        'pe_type': 'original',  # or 'sin'
        'use_residual': '[0,1,2,4,5]',

    }

    # Updating the out_name format

    # Updating the out_name format
    params['out_name'] = out_name

    # Loop through the parameters

    
    # Construct the output directory and other variables
    learning_rate = round(learning_rate, 8)
    output_directory = os.path.join(params['out_dir'], f"{params['out_name']}/addition_reverse_sd{params['general_seed']}_lr{learning_rate}_wu{warmup_iters}")
    wandb_run_name = f"addition_reverse_sd{params['general_seed']}_lr{learning_rate}_wu{warmup_iters}"
    wandb_project = params['out_name']

    # Construct the command using parameters from the dictionary
    command_params = {
        'use_pe': params['pe_type'],
        'use_residual': params['use_residual'],
        'max_iters': params['max_iters'],
        'lr_decay_iters': params['lr_decay_iters'],
        'layerwise_pe': layerwise_pe,
        'out_dir': output_directory,
        'wandb_run_name': wandb_run_name,
        'learning_rate': learning_rate,
        'warmup_iters': warmup_iters,
        'wandb_project': wandb_project
    }

    command = "python train.py pe_info/config2_pe/addition/reverse/jason_train_addition_bal.py "
    for key, value in command_params.items():
        command += f" --{key}={value}"
    print(command)
    os.system(command)

    output_files = glob.glob('./'+output_directory + '**/**/**.csv', recursive=True)
    output_files.sort(key=os.path.getctime, reverse=True)
    latest_file = output_files[0]
    df = pd.read_csv(latest_file)
    train_loss = df['train_loss'].values[-1]
    return train_loss



if __name__ == "__main__":
    out_dir = "./outputs"
    out_name = f"out3_tuning"
    os.makedirs(f"{out_dir}/{out_name}", exist_ok=True)
    storage = JournalStorage(
        JournalFileStorage(f"{out_dir}/{out_name}/tuning.log")
    )
    # try:
    #     study = optuna.load_study(
    #         storage=storage,
    #         study_name=out_name,
    #     )
    # except:
    study = optuna.create_study(
        direction="minimize",
        storage=storage,  # Specify the storage URL here.
        study_name=out_name,
        pruner=optuna.pruners.MedianPruner(),
        load_if_exists=True,
    )
    trial_function = partial(
        run_tuning, out_name = out_name
    )
    study.optimize(trial_function, n_trials=32, n_jobs=2,)
    print(study.best_params)