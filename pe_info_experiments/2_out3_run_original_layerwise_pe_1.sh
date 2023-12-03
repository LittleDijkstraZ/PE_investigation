#!/bin/bash


out_name='ouputs/out3'
pe_type='original'


#0. add a layerwise pe to the following no residual config
use_residual='[0,2,3,4,5]'
# layerwise_pe='[1]' # saved a bit
# layerwise_pe='[2]' # how about pe at 2 -- works better
# layerwise_pe='[3]' # how about pe at 3 -- even higher！
# layerwise_pe='[4]' # how about pe at 4 -- 

layerwise_pe='[1,2]' # ok, interestingly, when i was about tho type this, copilot read my mine : )

python train.py pe_info/config2_pe/addition/reverse/jason_train_addition_bal.py \
  --use_pe="$pe_type" --use_residual=$use_residual \
  --layerwise_pe=$layerwise_pe \
  --out_dir="$out_name/addition_reverse" \
  --wandb_run_name='addition_reverse' \
  --wandb_project="$out_name"


# 1. try also to rescue this settings
# use_residual='[0,1,2,3,4,]'
# layerwise_pe='[5]' # 



# python train.py pe_info/config2_pe/addition/reverse/jason_train_addition_bal.py \
#   --use_pe="$pe_type" --use_residual=$use_residual \
#   --layerwise_pe=$layerwise_pe \
#   --out_dir="$out_name/addition_reverse" \
#   --wandb_run_name='addition_reverse' \
#   --wandb_project="$out_name"