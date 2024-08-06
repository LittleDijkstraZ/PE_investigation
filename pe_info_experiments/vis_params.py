import torch
# -----------------------------------------------------------------------------
# I/O
out_dir = 'out'
resume_dir = None
resume_iter = False  # if True, resume from saved iter_num, otherwise resume from iter_num 0
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False  # if True, script exits right after the first eval
always_save_checkpoint = True  # if True, always save a checkpoint after each eval
init_from = 'scratch'  # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_entity = 'ssdd'
wandb_log = True  # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2'  # 'run' + str(time.time())
exp_name = 'default_exp_name'
# data
train_data_path = 'train_3digit_10000.txt'
gradient_accumulation_steps = 1  # used to simulate larger batch sizes
test_batch_size = 128
batch_size = 12  # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
val_data_path = 'val.bin'
multi_digit = False
num_digit = 5
binary = False
# using two data - data1 = text / data2 = addition
# use seperate text/add data for train/val (get_batch uses this to sample from two differernt datasets)
train_both = False
data_ratio = 0.2  # ratio of data_path2 compared with data_path1
train_data_path2 = 'train_addition.bin'  # only used when train_both = True
val_data_path2 = 'val_addition.bin'
# evaluation
eval_text = False  # if True get perplexity using eval_text_data_path
# directory to text data (.bin file) - ex. 'data/shakespeare_add_ar_mixed/val_text.bin'
eval_text_data_path = None
eval_addition = False  # if True compute test accuracy of "a+b="
start = "FILE:data/bal/test_10000.txt"
eval_addition_ar = False
start_ar = None
# use this to evaluate other operations (ex. train on operator '-' but evaluate on other_operator '+')
eval_other = False
start_other = None
other_operator = '+'
eval_addition_train = False
start_train = None
reverse_ab = False
reverse_c = False
zero_pad = False
algo_reason = False
add_space = False
# model
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2
bias = False  # do we use bias inside LayerNorm and Linear layers?
ckpt_path_name = 'ckpt.pt'
save_final = True
# adamw optimizer
learning_rate = 6e-4  # max learning rate
max_iters = 600000  # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True  # whether to decay the learning rate
warmup_iters = 2000  # how many steps to warm up for
lr_decay_iters = 600000  # should be ~= max_iters per Chinchilla
min_lr = None  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl'  # 'nccl', 'gloo', etc.
# system
device = 'cuda' if torch.cuda.is_available() else 'cpu' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
# 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
try:
    dtype = 'bfloat16' if torch.cuda.is_bf16_supported() else 'float16'
except:
    dtype = 'float16'
compile = True  # use PyTorch 2.0 to compile the model to be faster
use_flash = True
data_type = 'text'
dataset = 'bal'
operator = '+'  # can be '+', '-', '*', 'sin', 'sqrt'
data_shuffle = True
# data_format = 'algo_reasoning' # 'plain' or 'reverse' or 'algo_reasoning'
data_format = 'plain'  # 'plain' or 'reverse' or 'algo_reasoning'

# can be 'all_ascii_chars' or 'numbers_only' or 'custom_input_data'
vocabulary = 'all_ascii_chars'
meta_path_specified = False  # use saved meta_file (False if data_type='text')
eps = 0
tokenizer = 'char'  # by default, use char level tokenizer. but for pretrained models, use openai tokenizer eg: 'gpt2'

simple = False
random_A = False
random_C = False

use_lora = False  # use lora (from minLoRA)
print_interval = 2  # if we're using gpt-2 model, I want to see it prompted on text

general_seed = 1337
# general_seed = 1227
resume_metric_from_best = True
use_pe = 'original'
use_residual = True
no_att_residual = False
no_mlp_residual = False
layerwise_pe = False
permute = False
not_causal = False

causal_training = True