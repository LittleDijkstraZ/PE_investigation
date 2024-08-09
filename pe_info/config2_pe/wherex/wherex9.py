# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such


use_pe = 'original'
pe_status = '' if use_pe=='original' else f'_{use_pe}'
use_residual = True
residual_status = '' if use_residual else '_noresidual'


out_dir = f'outputs/wherex9{pe_status}{residual_status}'
eval_interval = 250 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10 # don't print too too often



# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = True # override via command line if you like
wandb_project = 'wherex9'
wandb_run_name = f'wherex9{pe_status}{residual_status}'

data_type='text'
data_format='plain'
operator = 'wherex'
dataset = 'wherex'
# batch_size = 256
# batch_size = 2048 # much larger batch size
block_size = 256 # context of up to 256 pwherexious characters
train_data_path = 'train_wherex9_10000.txt'
# val_data_path = 'val.bin'
ckpt_path_name = 'ckpt_10000.pt'
eval_addition = True
start = "FILE:data/wherex/test_wherex9_10000.txt"
eval_addition_train = True
# start_train = "FILE:data/one-sided-subtraction/plain/add_examples_10000_trainprompt.txt"

# baby GPT model :)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

beta2 = 0.99 # make a bit bigger because number of tokens per iter is small


device='cuda:0'

batch_size = 4096
max_iters = 2000 
lr_decay_iters = 2000 # make equal to max_iters usually
learning_rate = 0.000026441
warmup_iters = 400


# causal_training=False # we can train it non-causally as well
# non_causal_fix_length = 14 # for 5-digit wherex

# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model
