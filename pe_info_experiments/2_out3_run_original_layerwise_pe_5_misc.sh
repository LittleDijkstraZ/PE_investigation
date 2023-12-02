#!/bin/bash


# max_iters=7500
# lr_decay_iters=7500
# general_seed=888


max_iters=5000
lr_decay_iters=5000
general_seed=8


out_name="out3_iters7500"_"888"
pe_type='original'
# pe_type='sin'

echo "Running with general_seed: $out_name"


use_residual='[0,1,2,4,5]'
layerwise_pe_list=('[3,4]' '[2,3,4]' '[3,4,5]') # 

learning_rate=0.01

for layerwise_pe in "${layerwise_pe_list[@]}"
do
  echo "Running with layerwise_pe: $layerwise_pe"
  python train.py pe_info/config2_pe/addition/reverse/jason_train_addition_bal.py \
    --use_pe="$pe_type" --use_residual=$use_residual \
    --max_iters=$max_iters --lr_decay_iters=$lr_decay_iters \
    --general_seed=$general_seed \
    --layerwise_pe=$layerwise_pe \
    --out_dir="$out_name/addition_reverse_sd$general_seed"_"lr$learning_rate" \
    --wandb_run_name="addition_reverse_sd$general_seed"_"lr$learning_rate" \
    --learning_rate=$learning_rate \
    --wandb_project="$out_name"
done  