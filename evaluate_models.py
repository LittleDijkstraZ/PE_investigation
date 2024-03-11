"""
Sample from a trained model
"""
print('importing...')
import os
import pickle
from contextlib import nullcontext
print('importing...')
import torch
import tiktoken
# from model import GPTConfig, GPT
from tqdm import tqdm
import glob
import re
import numpy as np
print('importing...')
# from main_utils import load_trained_model, evaluate_addition, get_encode_decode, evaluate_text
from main_utils import *

print('finish import')
# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out-addition-char-pad' # ignored if init_from is not 'resume'
start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 1 # number of samples to draw
max_new_tokens = 4 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
ckpt_path_name = 'ckpt.pt'
dataset = 'shakespeare_addition_char'
batch_size = 64
analyze = False

train_data_path = 'train.bin'
val_data_path = 'val.bin'

device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
use_flash = True
data_type = 'binary' # 'binary' by default, can be 'text'
operator = '+' # can be '+', '-', '*', 'sin', 'sqrt'
data_shuffle = True
data_format = 'reverse'
vocabulary = 'all_ascii_chars' # can be 'all_ascii_chars' or 'numbers_only' or 'custom_input_data'
add_space = False
tokenizer = 'char' # by default, use char level tokenizer. but for pretrained models, use openai tokenizer eg: 'gpt2'
simple=False
random_A=False
random_C=False


wandb_log=True
wandb_entity = 'littledijkstraz'
wandb_project = 'addition-char'
wandb_run_name = 'num_train-mini-gpt-padded'

plot_sample_acc = True
eval_addition = True
eval_text = False
eval_text_data_path = 'data/shakespeare_addition_char/val_shakespeare.bin' 

print('configuring...')
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str, type(None)))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------
import yaml
import glob
glob_path = f'{out_dir}/**/config.yaml'
glob_path = glob_path.replace('[', '*')
glob_path = glob_path.replace(']', '*')
print(glob_path)
cfg_path = glob.glob(glob_path, recursive=True)[0]
config = yaml.load(open(cfg_path, 'r'), Loader=yaml.FullLoader)
for k in config:
    globals()[k] = config[k]
    
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# look for the meta pickle in case it is available in the dataset folder
print('dataset', dataset)
data_dir = os.path.join('data', dataset)
meta_path = os.path.join(data_dir, 'meta.pkl')
if not os.path.exists(meta_path):
    meta_path = None
    train_data_path = os.path.join(data_dir, train_data_path)
    # val_data = os.path.join(data_dir, val_data_path)
    train_data_list = get_data_list(train_data_path, operator=operator)
    val_data_list = get_data_list(filename=None, operator=operator) # get_data_list(val_data, operator='+')
    train_data_str = generate_data_str(train_data_list, operator=operator, format=data_format, train=True, shuffle=data_shuffle, add_space=add_space, simple=simple, random_A=random_A, random_C=random_C)
    val_data_str = generate_data_str(val_data_list, operator=operator, format=data_format, train=True, shuffle=data_shuffle, add_space=add_space, simple=simple, random_A=random_A, random_C=random_C)
    meta, meta_path, data_encoder, data_decoder = create_meta_file(vocabulary=vocabulary, input_data_str=train_data_str, tokenizer=tokenizer)

print('meta_path', meta_path)


encode, decode = get_encode_decode(meta_path)


# if wandb_log:
#     print('Logging to wandb')
#     import wandb
#     wandb.init(project=wandb_project, entity=wandb_entity, name=wandb_run_name, config=config)
ckpt_out_dir = out_dir.replace('[', '*').replace(']', '*')
# ckpt_list = glob.glob(ckpt_out_dir+'/ckpt_*_final.pt')
ckpt_list = glob.glob(ckpt_out_dir+'/ckpt_10000_acc.pt')
ckpt_list.sort(key=lambda test_string : list(map(int, re.findall(r'\d+', test_string)))[0])
print(ckpt_list)

def load_checkpoint(ckpt_path, model_config, model_type, device='cuda', return_config=False):
        # load ckpt into model
        checkpoint = torch.load(ckpt_path, map_location=device)

        model_args = checkpoint['model_args']
        # for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
                # model_args[k] = checkpoint_model_args[k]
        # for k in checkpoint_model_args:
        #         model_args[k] = checkpoint_model_args[k]
        # create the model
        original_gptconf = model_config(**model_args)
        gptconf = model_config(**model_args)
        model = model_type(original_gptconf)
        state_dict = checkpoint['model']
        # fix the keys of the state dictionary :(
        # honestly no idea how checkpoints sometimes get this prefix, have to debug more
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
                if k.startswith(unwanted_prefix):
                        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        if return_config:
                return model, gptconf
        else:
                return model
        
from pe_info.model_nope import GPTConfig as GPTConfig_nope, GPT as GPT_nope

for ckpt_path_name in ckpt_list:
    # load model
    print(f'loading: {ckpt_path_name}')
    num_train = int(''.join(x for x in ckpt_path_name if x.isdigit()))

    # checkpoint = torch.load(ckpt_path_name, map_location=device)
    # model = load_trained_model(config, checkpoint)
    model, gptconfig = load_checkpoint(ckpt_path_name, GPTConfig_nope, GPT_nope, device='cuda', return_config=True)

    model.eval()
    model.to(device)
    if compile:
        model = torch.compile(model) # requires PyTorch 2.0 (optional)

    # evaluate addition performance
    if eval_addition:
        test_accuracy, accuracy_dict = evaluate_addition(config, model, ctx, encode, decode, verbose=True, analyze=analyze)
    if eval_text:
        eval_text_data = np.memmap(eval_text_data_path, dtype=np.uint16, mode='r')
        ppl = evaluate_text(config, model, eval_text_data, ctx)
    if wandb_log:
        wandb_dict={"num_train_samples": num_train, 
        "test_accuracy": test_accuracy if eval_addition else None,
        "test_perplexity": ppl if eval_text else None
        }
        wandb_dict.update(accuracy_dict)
        print(wandb_dict)
        # wandb.log(wandb_dict)