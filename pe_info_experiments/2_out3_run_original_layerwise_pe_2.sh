#!/bin/bash



max_iters=7500
lr_decay_iters=7500
general_seed=888
out_name="out3_iters$max_iters"_"$general_seed"
pe_type='original'
echo "Running with general_seed: $out_name"


# 3. maybe this one
use_residual='[1,2,3,4,5]'
layerwise_pe_list=(False '[0]' '[1]' '[2]' '[3]' '[4]' '[5]' True) # true means do it for all

for layerwise_pe in "${layerwise_pe_list[@]}"
do 
  echo "Running with layerwise_pe: $layerwise_pe"
  python train.py pe_info/config2_pe/addition/reverse/jason_train_addition_bal.py \
    --use_pe="$pe_type" --use_residual=$use_residual \
    --max_iters=$max_iters --lr_decay_iters=$lr_decay_iters \
    --general_seed=$general_seed \
    --layerwise_pe=$layerwise_pe \
    --out_dir="$out_name/addition_reverse" \
    --wandb_run_name='addition_reverse' \
    --wandb_project="$out_name"
done



# use_residual='[0,2,3,4,5,6]'
# n_layer=7
# python train.py pe_info/config2_pe/addition/reverse/jason_train_addition_bal.py \
#   --use_pe="$pe_type" --use_residual=$use_residual \
#   --max_iters=$max_iters --lr_decay_iters=$lr_decay_iters \
#   --n_layer=$n_layer \
#   --general_seed=$general_seed \
#   --out_dir="$out_name/addition_reverse" \
#   --wandb_run_name='addition_reverse' \
#   --wandb_project="$out_name"


max_iters=7500
lr_decay_iters=7500
out_name="out3_iters$max_iters"
pe_type='original'
echo "Running with general_seed: $out_name"


use_residual='[0,1,2,4,5]'
layerwise_pe_list=('[3,4]' '[2,4]' '[2,3,4]') # 

for layerwise_pe in "${layerwise_pe_list[@]}"
do
  echo "Running with layerwise_pe: $layerwise_pe"
  python train.py pe_info/config2_pe/addition/reverse/jason_train_addition_bal.py \
    --use_pe="$pe_type" --use_residual=$use_residual \
    --max_iters=$max_iters --lr_decay_iters=$lr_decay_iters \
    --general_seed=$general_seed \
    --layerwise_pe=$layerwise_pe \
    --out_dir="$out_name/addition_reverse" \
    --wandb_run_name='addition_reverse' \
    --wandb_project="$out_name"
done  