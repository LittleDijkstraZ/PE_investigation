#!/bin/bash

cd "./teaching_arithmetic_pe"

out_name='out3'

# python train.py pe_info/config2_pe/addition/reverse/jason_train_addition_bal.py \
#   --use_pe='nope' --use_residual=[0,1,2,3,4,5] \
#   --out_dir="$out_name/addition_reverse" \
#   --wandb_run_name='addition_reverse' \
#   --wandb_project="$out_name"

python train.py pe_info/config2_pe/addition/reverse/jason_train_addition_bal.py \
  --use_pe='nope' --use_residual=[1,2,3,4,5] \
  --out_dir="$out_name/addition_reverse" \
  --wandb_run_name='addition_reverse' \
  --wandb_project="$out_name"

python train.py pe_info/config2_pe/addition/reverse/jason_train_addition_bal.py \
  --use_pe='nope' --use_residual=[2,3,4,5] \
  --out_dir="$out_name/addition_reverse" \
  --wandb_run_name='addition_reverse' \
  --wandb_project="$out_name"

python train.py pe_info/config2_pe/addition/reverse/jason_train_addition_bal.py \
  --use_pe='nope' --use_residual=[0,1,2,3,4] \
  --out_dir="$out_name/addition_reverse" \
  --wandb_run_name='addition_reverse' \
  --wandb_project="$out_name"

python train.py pe_info/config2_pe/addition/reverse/jason_train_addition_bal.py \
  --use_pe='nope' --use_residual=[0,1,2,3] \
  --out_dir="$out_name/addition_reverse" \
  --wandb_run_name='addition_reverse' \
  --wandb_project="$out_name"

# python train.py pe_info/config2_pe/addition/reverse/jason_train_addition_bal.py \
#   --use_pe='nope' --use_residual=[2,3,4,5] \
#   --out_dir='out2/addition_reverse_bugfixed' \
#   --wandb_run_name='addition_reverse_bugfixed'

# python train.py pe_info/config2_pe/addition/reverse/jason_train_addition_bal.py \
#   --use_pe='nope' --use_residual=[0,2,3,4,5] \
#   --out_dir='out2/addition_reverse_bugfixed' \
#   --wandb_run_name='addition_reverse_bugfixed'

# python train.py pe_info/config2_pe/addition/reverse/jason_train_addition_bal.py \
#   --use_pe='nope' --use_residual=[0,3,4,5] \
#   --out_dir='out2/addition_reverse_bugfixed' \
#   --wandb_run_name='addition_reverse_bugfixed'

