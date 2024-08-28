
import torch

from main_utils import *
from contextlib import nullcontext


def load_config_and_data(path='./pe_info/config2_pe/rev/rev16.py', 
                         start=None,
                         dtype=None,
                         data_type=None,
                         dataset=None,
                         train_data_path=None,
                         train_data_path2=None,
                         val_data_path=None,
                         val_data_path2=None,
                         eval_text=False,
                         eval_text_data_path=None,
                         train_both=False,
                         data_format=None,
                         reverse_c=False,
                         algo_reason=False,
                         operator=None,
                         data_shuffle=False,
                         add_space=False,
                         simple=False,
                         random_A=False,
                         random_C=False,
                         vocabulary=None,
                         tokenizer_type=None,
                         eval_addition_train=False,
                         start_train=None,
                         **kwargs):

    val_data_path = start.split(':')[-1] # get the test data path

    # val_data_path = start
    print(val_data_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
    val_data_path = start.split(':')[-1] # get the test data path

    # for later use in torch.autocast
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    # note: float16 data type will automatically use a GradScaler
    ptdtype = {'float32': torch.float32,
            'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(
        device_type=device_type, dtype=ptdtype)

    # poor man's data loader
    if data_type == 'binary':
        data_dir = os.path.join('data', dataset)
        train_data = np.memmap(os.path.join(
            data_dir, train_data_path), dtype=np.uint16, mode='r')
        val_data = np.memmap(os.path.join(
            data_dir, val_data_path), dtype=np.uint16, mode='r')
        if train_both:
            train_data2 = np.memmap(os.path.join(
                data_dir, train_data_path2), dtype=np.uint16, mode='r')
            val_data2 = np.memmap(os.path.join(
                data_dir, val_data_path2), dtype=np.uint16, mode='r')
        if eval_text:
            if eval_text_data_path is None:
                print(
                    'eval_text_data_path is None!!! No binary file to evaluate perplexity on.')
            eval_text_data = np.memmap(
                eval_text_data_path, dtype=np.uint16, mode='r')
        # test_data_str = None # test_data for addition testing will be handled with "start"
        meta_path = None
    else:
        # check for data_format
        if data_type == 'text':
            if ('reverse' in data_format and not reverse_c) or (reverse_c and 'reverse' not in data_format):
                raise ValueError(
                    'reverse_c must be True for data_format == "reverse"')
            elif (data_format == 'algo_reasoning' and not algo_reason) or (algo_reason and data_format != 'algo_reasoning'):
                raise ValueError(
                    'algo_reason must be True for data_format == "algo_reasoning"')
        meta_path_specified = False

        data_dir = os.path.join('data', dataset)
        train_data_path = os.path.join(data_dir, train_data_path)
        # val_data = os.path.join(data_dir, val_data_path)
        train_data_list = get_data_list(
            train_data_path, operator=operator)  # a list of (x, y, op)
        val_data_list = get_data_list(filename=val_data_path, operator=operator)
        train_data_str = generate_data_str(train_data_list, operator=operator, format=data_format, train=True,
                                        shuffle=data_shuffle, add_space=add_space, simple=simple, random_A=random_A, random_C=random_C)
        val_data_str = generate_data_str(val_data_list, operator=operator, format=data_format, train=True,
                                        shuffle=data_shuffle, add_space=add_space, simple=simple, random_A=random_A, random_C=random_C)
        meta, meta_path, data_encoder, data_decoder = create_meta_file(
            vocabulary=vocabulary, input_data_str=train_data_str, tokenizer=tokenizer_type)
        

        meta_vocab_size = meta['vocab_size']
        train_data = data_encoder(train_data_str)
        val_data = data_encoder(val_data_str)
        if eval_addition_train and start_train is None:
            # specify the start_train to be our train data file
            start_train = f"FILE:{train_data_path}"

        if train_both:
            # This is for the case where we use two different datasets for training
            # we sample from both with a specified ratio - data_ratio
            # TODO: let's leave this here for now.
            train_data2 = np.memmap(os.path.join(
                data_dir, train_data_path2), dtype=np.uint16, mode='r')
            val_data2 = np.memmap(os.path.join(
                data_dir, val_data_path2), dtype=np.uint16, mode='r')

        if eval_text:
            # eval_text_data = np.memmap(eval_text_data_path, dtype=np.uint16, mode='r')
            text_data_list = get_data_list(eval_text_data_path, operator='text')
            text_data_str = generate_data_str(
                text_data_list, operator='text', format=data_format, train=False, shuffle=False)
            eval_text_data = data_encoder(text_data_str)

    if meta_path_specified:
        # attempt to derive vocab_size from the dataset
        meta_path = os.path.join(data_dir, 'meta.pkl')
        meta_vocab_size = None
        if os.path.exists(meta_path):
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
            meta_vocab_size = meta['vocab_size']
            print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")
        else:
            meta_path = None

    encode, decode = get_encode_decode(meta_path, tokenizer=tokenizer_type)

    return {
        'encode': encode,
        'decode': decode,
        'train_data': train_data,
        'val_data': val_data,
        'device': device,
        'ctx': ctx,
        'meta': meta,
        'meta_path': meta_path,
        'tokenizer_type': tokenizer_type,
        'data_encoder': data_encoder,
        'data_decoder': data_decoder,
    }