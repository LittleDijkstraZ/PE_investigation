# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such



use_pe = 'original'
pe_status = '' if use_pe=='original' else f'_{use_pe}'
use_residual = True
residual_status = '' if use_residual else '_noresidual'


out_dir = f'outputs/modclean{pe_status}{residual_status}'
eval_interval = 250 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10 # don't print too too often



# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = True # override via command line if you like
wandb_project = 'modclean'
wandb_run_name = f'modclean{pe_status}{residual_status}'

data_type='text'
data_format='plain'
operator = 'modclean'
dataset = 'modclean'

block_size = 256 # context of up to 256 previous characters
train_data_path = 'train_modclean_10000.txt'
# val_data_path = 'val.bin'
ckpt_path_name = 'ckpt_10000.pt'
eval_addition = True
start = "FILE:data/modclean/test_modclean_10000.txt"
eval_addition_train = True
# start_train = "FILE:data/one-sided-subtraction/plain/add_examples_10000_trainprompt.txt"

# baby GPT model :)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

# learning_rate = 1e-3 # with baby networks can afford to go a bit higher
# max_iters = 5000
# lr_decay_iters = 5000 # make equal to max_iters usually
# beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

batch_size = 4096
max_iters = 2000 
learning_rate = 0.000026441
warmup_iters = 400

device='cuda:0'
# causal_training=False # we can train it non-causally as well
# non_causal_fix_length = 14 # for 5-digit modclean

# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model
