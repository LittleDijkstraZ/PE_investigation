#!/bin/bash

cd "./teaching_arithmetic_pe"

out_name='out3_control'
pe_type='original'
general_seed=888
# pe_type='nope'


#0. add a layerwise pe to the following no residual config
# n_layer=4 # micmic use_residual='[0,2,3,4,5]'



# python train.py pe_info/config2_pe/addition/reverse/jason_train_addition_bal.py \
#   --use_pe="$pe_type" \
#   --n_layer=$n_layer \
#   --out_dir="$out_name/addition_reverse" \
#   --wandb_run_name='addition_reverse' \
#   --wandb_project="$out_name"



python train.py pe_info/config2_pe/addition/reverse/jason_train_addition_bal.py \
  --use_pe="$pe_type" \
  --general_seed=$general_seed \
  --out_dir="$out_name/addition_reverse" \
  --wandb_run_name='addition_reverse' \
  --wandb_project="$out_name"
