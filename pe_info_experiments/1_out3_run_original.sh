#!/bin/bash



out_name='out3'
pe_type='original'

# python train.py pe_info/config2_pe/addition/reverse/jason_train_addition_bal.py \
#   --use_pe="$pe_type" --use_residual=[0,1,2,3,4,5] \
#   --out_dir="$out_name/addition_reverse" \
#   --wandb_run_name='addition_reverse' \
#   --wandb_project="$out_name"

# Define an array of use_residual values
# residual_values=('[1,2,3,4,5]' '[0,2,3,4,5]' '[0,1,3,4,5]' '[0,1,2,4,5]' '[0,1,2,3,5]' '[0,1,2,3,4]')
# residual_values=('[0,1,2,3,4]')
residual_values=('[0,2,3,4,5]')



# Loop over each value in the array
for use_residual in "${residual_values[@]}"
do
    echo "Running with use_residual: $use_residual"

    # Call your Python script with the current use_residual value
    python train.py pe_info/config2_pe/addition/reverse/jason_train_addition_bal.py \
      --use_pe="$pe_type" --use_residual=$use_residual \
      --out_dir="$out_name/addition_reverse" \
      --wandb_run_name='addition_reverse' \
      --wandb_project="$out_name"

    # Optional: Add a wait or sleep here if needed
    # sleep 1
done

# python train.py pe_info/config2_pe/addition/reverse/jason_train_addition_bal.py \
#   --use_pe="$pe_type" --use_residual=[2,3,4,5] \
#   --out_dir="$out_name/addition_reverse" \
#   --wandb_run_name='addition_reverse' \
#   --wandb_project="$out_name"

# python train.py pe_info/config2_pe/addition/reverse/jason_train_addition_bal.py \
#   --use_pe="$pe_type" --use_residual=[0,1,2,3] \
#   --out_dir="$out_name/addition_reverse" \
#   --wandb_run_name='addition_reverse' \
#   --wandb_project="$out_name"



# python train.py pe_info/config2_pe/addition/reverse/jason_train_addition_bal.py \
#   --use_pe="$pe_type" --use_residual=[2,3,4,5] \
#   --out_dir='out2/addition_reverse_bugfixed' \
#   --wandb_run_name='addition_reverse_bugfixed'

# python train.py pe_info/config2_pe/addition/reverse/jason_train_addition_bal.py \
#   --use_pe="$pe_type" --use_residual=[0,2,3,4,5] \
#   --out_dir='out2/addition_reverse_bugfixed' \
#   --wandb_run_name='addition_reverse_bugfixed'

# python train.py pe_info/config2_pe/addition/reverse/jason_train_addition_bal.py \
#   --use_pe="$pe_type" --use_residual=[0,3,4,5] \
#   --out_dir='out2/addition_reverse_bugfixed' \
#   --wandb_run_name='addition_reverse_bugfixed'

