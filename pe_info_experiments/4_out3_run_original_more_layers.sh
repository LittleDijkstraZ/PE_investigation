#!/bin/bash

out_dir='outputs'

general_seed=888
# set out_name to out 3 if general seed is 1337
if [ $general_seed -eq 1337 ]
then
  out_name='out3'
else
  out_name='out3_'$general_seed
fi

pe_type='original'


##1. add more layers to rescue [0,2,3,4,5]

# # not enough, maybe retrain with a different random seed
# use_residual='[0,2,3,4,5,6]'
# n_layer=7
# python train.py pe_info/config2_pe/addition/reverse/jason_train_addition_bal.py \
#       --use_pe="$pe_type" --use_residual=$use_residual \
#       --n_layer=$n_layer \
#       --general_seed=$general_seed \
#       --out_dir="$out_name/addition_reverse" \
#       --wandb_run_name='addition_reverse' \
#       --wandb_project="$out_name"

# ## just enough
# use_residual='[0,2,3,4,5,6,7]'
# n_layer=8
# python train.py pe_info/config2_pe/addition/reverse/jason_train_addition_bal.py \
#       --use_pe="$pe_type" --use_residual=$use_residual \
#       --n_layer=$n_layer \
#       --general_seed=$general_seed \
#       --out_dir="$out_name/addition_reverse" \
#       --wandb_run_name='addition_reverse' \
#       --wandb_project="$out_name"

## 2. 9 layers
### how bout? didnt work
# use_residual='[0,2,3,4,5,6,7,8]'
# n_layer=9
# python train.py pe_info/config2_pe/addition/reverse/jason_train_addition_bal.py \
#       --use_pe="$pe_type" --use_residual=$use_residual \
#       --n_layer=$n_layer \
#       --general_seed=$general_seed \
#       --out_dir="$out_name/addition_reverse" \
#       --wandb_run_name='addition_reverse' \
#       --wandb_project="$out_name"

# ## play a trick? more missings : ) didnt work
# use_residual='[2,3,4,5,6,7,8]' 
# n_layer=9
# python train.py pe_info/config2_pe/addition/reverse/jason_train_addition_bal.py \
#       --use_pe="$pe_type" --use_residual=$use_residual \
#       --n_layer=$n_layer \
#       --general_seed=$general_seed \
#       --out_dir="$out_name/addition_reverse" \
#       --wandb_run_name='addition_reverse' \
#       --wandb_project="$out_name"

## 3. more harsh 8 layers
# use_residual='[2,3,4,5,6,7]' 
# n_layer=8
# python train.py pe_info/config2_pe/addition/reverse/jason_train_addition_bal.py \
#       --use_pe="$pe_type" --use_residual=$use_residual \
#       --n_layer=$n_layer \
#       --general_seed=$general_seed \
#       --out_dir="$out_name/addition_reverse" \
#       --wandb_run_name='addition_reverse' \
#       --wandb_project="$out_name"

# use_residual='[0,3,4,5,6,7]' 
# n_layer=8
# python train.py pe_info/config2_pe/addition/reverse/jason_train_addition_bal.py \
#       --use_pe="$pe_type" --use_residual=$use_residual \
#       --n_layer=$n_layer \
#       --general_seed=$general_seed \
#       --out_dir="$out_dir/$out_name/addition_reverse" \
#       --wandb_run_name='addition_reverse' \
#       --wandb_project="$out_name"

## 4. repeat 9 layers with different learning rate