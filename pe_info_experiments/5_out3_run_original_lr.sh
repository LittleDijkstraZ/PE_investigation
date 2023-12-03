#!/bin/bash


# max_iters=7500
# lr_decay_iters=7500
# general_seed=888


max_iters=5000
lr_decay_iters=5000
general_seed=888

out_dir='outputs'
out_name="out3_iters_$max_iters"_"$general_seed"
pe_type='original'
# pe_type='sin'

echo "Running with general_seed: $out_name"


use_residual='[0,1,2,4,5]'
layerwise_pe_list=('[3,4,5]') # 

# learning_rate_list=(0.03 0.01 0.005 0.001 0.0001)
# learning_rate_list=(0.01 0.005 0.001 0.0001)

## okey, so these are the sets that reach convergence!
learning_rate_list=(0.0001)
warmup_iters_list=(500)

for warmup_iters in "${warmup_iters_list[@]}"
do
  echo "Running with learning_rate: $warmup_iters"
  for learning_rate in "${learning_rate_list[@]}"
  do
    echo "Running with learning_rate: $learning_rate"
    for layerwise_pe in "${layerwise_pe_list[@]}"
    do
      echo "Running with layerwise_pe: $layerwise_pe"
      python train.py pe_info/config2_pe/addition/reverse/jason_train_addition_bal.py \
        --use_pe="$pe_type" --use_residual=$use_residual \
        --max_iters=$max_iters --lr_decay_iters=$lr_decay_iters \
        --layerwise_pe=$layerwise_pe \
        --out_dir="$out_dir/$out_name/addition_reverse_sd$general_seed"_"lr$learning_rate"_"wu$warmup_iters" \
        --wandb_run_name="addition_reverse_sd$general_seed"_"lr$learning_rate"_"wu$warmup_iters" \
        --learning_rate=$learning_rate \
        --warmup_iters=$warmup_iters \
        --wandb_project="$out_name"
    done  
  done
done
