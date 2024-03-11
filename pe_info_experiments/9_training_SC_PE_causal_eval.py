# Define the parameters in a dictionary
import os
import glob
import pandas as pd
from functools import partial


def run_eval(out_dir):
    # layerwise_pe, use_residual = args

    # pe_type='original'


    command = "python3 evaluate_models.py pe_info/config2_pe/addition/reverse/jason_train_addition_bal.py "
    # for key, value in command_params.items():
    command += f" --out_dir=\"{out_dir}\""
    print(command)
    os.system(command)




if __name__ == "__main__":
    out_dir = "./outputs/nope_residual_exp/addition_reverse_sd111_T2401311659_nope_res=[1, 2, 3, 4, 5]"
    from multiprocessing import Pool
    from functools import partial
    

    pool = Pool(1)
    func = run_eval
    args = [ out_dir ]
    # for arg in args:
        # func(arg)
    pool.map(func, args)
    pool.close()