# before changing get_batch function() to use data_ratio for sampling between two different dataset (data1=text, data2=add)
# we created seperate dataset called 'train_all', 'val_all', consisting of shakespeare & addition dataset

# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = 'out-jt-shakespeare-addition-char-pad'
eval_interval = 250 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = True # override via command line if you like
wandb_project = 'jt-shakespeare-addition-char'
wandb_run_name = 'mini-gpt-padded'

dataset = 'shakespeare_addition_char'
train_data_path='train_all.bin' #'train_addition.bin'
val_data_path='val_all.bin' #'val_addition.bin'
batch_size = 256
block_size = 256 # context of up to 256 previous characters
init_from = 'scratch'
ckpt_path_name = 'ckpt.pt'
eval_addition = True
start = "FILE:prompt/prompt_addition_pad_test_0.0001.txt"
eval_text = True
eval_text_data_path = 'data/shakespeare_addition_char/val_shakespeare.bin' 

# baby GPT model :)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

learning_rate = 1e-3 # with baby networks can afford to go a bit higher
max_iters = 20000
lr_decay_iters = 20000 # make equal to max_iters usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 0 # 100 # not super necessary potentially

device='cuda:0'

# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model