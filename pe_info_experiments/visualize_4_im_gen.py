


fixed_length_dict = {
    'add':9,
    'rev16':22,
    'order':19,
    'wherex7':17,
    'wherex9':19,
    'identify':22,
}

total_samples = 1024
sample_num = 1024

equal_distancing_exp = False
use_1_sample = False
model_init = False
if model_init:
    print("model init")
    init_additional_config = {
        'n_head':1,
    }
else:
    init_additional_config = {}

ablation_config = {
    "V_out": False,
    "no_K": False,
    "no_V": False,
    "no_Q": False,
    "shrink_x": False,
    # "c_proj": True,
    # only one at a time
    "func_config": {
        0: None,
        1:"nodiv", 
        2:"softmax", 
        3:"abs", 
        4:"relu", 
        5:"divsum", 
        6:"gelu",
        7:"gelu_divabs",
        8:"sigmoid",
        9:"same"}[0],
}

if __name__ == '__main__':
    import sys
    import os

    # sys.path.append('../')
    # os.chdir('../')
    sys.path.append('./')
    sys.path.append('../')

    import functools

    from sqlalchemy import func
    import torch
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.distributed import init_process_group, destroy_process_group

    from IPython.utils import io
    from tqdm.auto import tqdm

    import glob
    import yaml
    import re


    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics.pairwise import cosine_similarity

    import pe_info
    from pe_info.model_nope import GPTConfig as GPTConfig_nope, GPT as GPT_nope
    from main_utils import *
    from vis_utils import get_batch_original 
    from vis_utils import get_corr as get_corr_original
    from vis_utils import load_checkpoint, generate_output, set_seed, get_PE_tendency, generate_tendency_map, load_config_and_data


    

    # os.chdir('../')

    task_config_dict = {
        'add': './pe_info/config2_pe/addition/reverse/jason_train_addition_bal.py',
        'order' : './pe_info/config2_pe/order/jason_train_addition_bal.py',
        'rev16' : './pe_info/config2_pe/rev/rev16.py',
        'wherex9' : './pe_info/config2_pe/wherex/wherex9.py',

    }
        

    for task_type in task_config_dict:
        try:
            exec(open('./pe_info_experiments/vis_params.py').read())
            exec(open(task_config_dict[task_type]).read())
            config_keys = [k for k, v in globals().items() if not k.startswith(
                        '_') and isinstance(v, (int, float, bool, str, type(None)))]
            config = {k: globals()[k] for k in config_keys}
            basic_variables_dict = load_config_and_data(path=task_config_dict[task_type], **config)

            train_data   = basic_variables_dict['train_data']
            val_data     = basic_variables_dict['val_data']
            data_encoder = basic_variables_dict['data_encoder']
            data_decoder = basic_variables_dict['data_decoder']
            device = basic_variables_dict['device']
            meta_path = basic_variables_dict['meta_path']
            tokenizer_type = basic_variables_dict['tokenizer_type']
            answer_length = 1

            encode, decode = get_encode_decode(meta_path, tokenizer=tokenizer_type)

            get_batch = functools.partial(get_batch_original, train_data=train_data, val_data=val_data, data_encoder=data_encoder,)
        

            exp_list = glob.glob('./outputs_ref_*/*/*') 
            exp_list = [p for p in exp_list if task_type in p]
            exp_list = [[x, x.split('/')[-1]] for x in exp_list] 
            print(len(exp_list))

            fixed_length = fixed_length_dict[task_type]

            all_level_input_act_list = []
            all_out_list = []

            rand_perm = False
            if rand_perm:
                print("permutation added")


            small_dim = False
            if small_dim:
                print("small dim")

            for k in ablation_config:
                if ablation_config[k]:
                    if k == "func_config":
                        print(f"{k}: {ablation_config[k]}")
                    else:
                        print(f"{k} is on")

            model_list = []
            useful_name_list = []
            for idx, (config_dir, model_config_fold) in enumerate(tqdm(exp_list)):
                glob_dir = config_dir.replace("[", "*").replace("]", "*")
                try:
                    yaml_path = glob.glob(f"{glob_dir}/**/config.yaml")[0]
                    csv_path = glob.glob(f"{glob_dir}/**/result.csv")[0]
                    revised_glob_dir = "/".join(yaml_path.split("/")[:-2])
                    exp_list[idx][0] = revised_glob_dir
                    exp_list[idx][1] = revised_glob_dir.split("/")[-1]

                    config_dir = "/".join(yaml_path.split("/")[:-2])
                    with open(yaml_path) as f:
                        config_dict = yaml.load(f, Loader=yaml.FullLoader)
                    df = pd.read_csv(csv_path)
                    
                    # ckpt = f"{config_dir}/ckpt_10000_acc.pt"
                    glob_dir = config_dir.replace("[", "*").replace("]", "*")
                    try:
                        all_ckpts = sorted(glob.glob(f"{glob_dir}/ckpt_*.pt"), key=lambda x: int(x.split('_')[-1].split('.')[0]))
                    except:
                        all_ckpts = [gdir for gdir in glob.glob(f"{glob_dir}/ckpt_*.pt") if '_acc' in gdir]
                    # ckpt = f"{config_dir}/ckpt_2000_acc.pt"
                
                    gptconfig = load_checkpoint(all_ckpts[0], GPTConfig_nope, GPT_nope, device='cuda', return_config=True, return_model=False)

                    if gptconfig.n_embd != 384: continue
                    if gptconfig.n_layer != 6 : continue
                    
                    # add the initialized model
                    set_seed(np.random.randint(0,200))
                    for init_scheme in [
                                        'default',
                                        # # "('normal',0,0.002)", 
                                        # "('normal',0,0.02)",
                                        # # "('normal',0,0.2)",
                                        # # "('normal',0,2)", 
                                        # # "('normal',4,0.02)",
                                        # # "('normal',8,0.02)",
                                        # # "('normal',100,0.02)",                        
                                        # '("uniform",-0.02,0.02)', 
                                        # '("uniform",-0.001,0.001)', 

                                        # '("uniform",-1,1)', 
                                        # '("uniform",-10,10)', 

                                        # '(xavier,1)'
                                        ]:

                        ckpt = all_ckpts[0]
                        with io.capture_output() as captured:
                            model, gptconfig = load_checkpoint(
                                ckpt,
                                GPTConfig_nope,
                                GPT_nope,
                                device="cuda" if torch.cuda.is_available() else "cpu",
                                return_config=True,
                                init=True,
                                init_additional_config={
                                    'init_scheme': init_scheme,
                                },
                            )

                        model.eval()
                        model.to(device)
                        model_list.append(model)
                        try:
                            iter_num = int(ckpt.split('_')[-1].split('.')[0])
                            convergence = 0
                            cur_model_name = 'acc='+str(int(convergence))+ '_' \
                                + '/'.join(ckpt.split('/')[3:]).replace(str(iter_num), '0') \
                                + f'_nembd={gptconfig.n_embd}_nlayers={gptconfig.n_layer}' + f'_{init_scheme}'
                        except:
                            cur_model_name = 'acc=0'+ '_' \
                                + '/'.join(ckpt.split('/')[3:]) \
                                + f'_nembd={gptconfig.n_embd}_nlayers={gptconfig.n_layer}' + f'_Init_{init_scheme}'
                        useful_name_list.append(cur_model_name)
                        
                    # for ckpt in all_ckpts[len(all_ckpts)//2:len(all_ckpts)//2+1] + all_ckpts[-1:]:

                    '''get trained model results'''
                    # for ckpt in all_ckpts[-1:]:

                    for ckpt in all_ckpts:

                        with io.capture_output() as captured:
                            model, gptconfig = load_checkpoint(
                                ckpt,
                                GPTConfig_nope,
                                GPT_nope,
                                device="cuda" if torch.cuda.is_available() else "cpu",
                                return_config=True,
                                init=model_init,
                                init_additional_config=init_additional_config,
                            )
                            if small_dim:
                                gptconfig.n_head = 1
                                # gptconfig.not_causal = True
                                gptconfig.n_embd = 16
                                model = GPT_nope(gptconfig)
                                            
                            model.eval()
                            model.to(device)
                            model_list.append(model)
                            try:
                                iter_num = int(ckpt.split('_')[-1].split('.')[0])
                                convergence = df.loc[df['iter']==iter_num, 'test_acc'].values[0]
                            
                            except:
                                convergence = df.loc[:, 'test_acc'].values.max()
                            cur_model_name = 'acc='+str(int(convergence))+ '_' \
                                    + '/'.join(ckpt.split('/')[3:]) \
                                    + f'_nembd={gptconfig.n_embd}_nlayers={gptconfig.n_layer}' + '_trained'
                            useful_name_list.append(cur_model_name)

                except (ValueError, IndexError) as e:
                    print(f"no model {glob_dir}")
                    print(e)
                    continue

            n_embd = model_list[0].config.n_embd
            equation_num = 1
            causal_training = True # for add3
            total_tokens = total_samples * 2 * fixed_length * equation_num
            diviser = 256 if causal_training else fixed_length
            samples_to_generate = total_tokens//fixed_length + total_tokens%fixed_length
            fixed_length = equation_num * fixed_length
            print('causal_training is', causal_training)


            X = get_batch(split="valid", batch_size=samples_to_generate)[0]
            X = "\n".join([decode(X[i].tolist()) for i in range(X.shape[0])])[:total_tokens] # truncate x to lower computation
            equations = X.split('\n')
            X_n = []
            for eidx in range(0, len(equations), equation_num):
                full_eqn = '\n'.join(equations[eidx:eidx+equation_num])
                # print(full_eqn)
                if task_type=='add' and not full_eqn.startswith('$'): continue
                if len(full_eqn)<fixed_length:
                    if len(full_eqn) < fixed_length//4: continue
                    full_eqn = '\n' * (fixed_length-len(full_eqn))  + full_eqn
                full_eqn = full_eqn[:fixed_length]
                X_n.append(full_eqn)
            X_n = X_n[:total_samples]
            X = torch.tensor(list(map(lambda x: encode(x), X_n))).long()
            print(X.shape, X.type())

            # X_n = np.array(list(X[:len(X) // fixed_length * fixed_length])).reshape(-1, fixed_length)

            if rand_perm: # shuffle in input sequence
                for xidx in range(X.shape[0]):
                    X[xidx] = X[xidx, torch.randperm(X[xidx].shape[0])]

            X = X.to(device)
            if use_1_sample:
                X = X[:1]

            for level in range(0, 13):
                level = level - 1
                input_act1_list = [list() for _ in range(len(model_list))]
                out_list = [list() for _ in range(len(model_list))]

                for midx, model in enumerate(model_list):
                    if len(model.transformer.h)<=level:
                        continue

                    activation = {}

                    def getActivation(name):
                        # the hook signature
                        def hook(model, input, output):
                            activation[name] = output.detach()

                        return hook
                

                    # register forward hooks on the layers of choice
                    if level >= 0:
                        # h1 = model.transformer.h[level].register_forward_hook(
                        #     getActivation(f"layer_{level}")
                        # )
                        # if hook_location == 'pre_attn':
                        # h1 = model.transformer.h[level].attn.pre_att_identity.register_forward_hook(
                        #     getActivation(f"layer_{level}")
                        # )
                        # elif hook_location == 'post_attn':
                        h1 = model.transformer.h[level].attn.identity.register_forward_hook(
                            getActivation(f"layer_{level}")
                        )

                    else:
                        # if 'nope' in exp_list[midx][0]:
                        # h1 = model.transformer.drop.register_forward_hook(
                        #     getActivation(f"layer_{level}")
                        # )
                        h1 = model.transformer.drop.register_forward_hook(
                            getActivation(f"layer_{level}")
                        )
                        # else:
                        #     h1 = model.transformer.wte.register_forward_hook(
                        #         getActivation(f"layer_{level}")
                        #     )


                    h2 = model.transformer.ln_f.register_forward_hook(
                        getActivation("x_out")
                    )

                    with torch.no_grad():
                        out0 = model(
                            X, equal_distancing_exp=equal_distancing_exp, ablation_config=ablation_config,
                        )[0]
            
                    h1.remove()
                    h2.remove()

                    acts = activation[f"layer_{level}"].detach().cpu().numpy()
                    
                    input_act1_list[midx].append(acts)
                    out_arg = out0.argmax(-1).detach().cpu().numpy()
                    decoded_out = list(decode(out_arg.flatten()))
                    
                    out_list[midx] = decoded_out

                for hidx in range(len(input_act1_list)):
                    if len(input_act1_list[hidx]) == 0:
                        # input_act1_list[hidx] 
                        pass
                    else:
                        cur_input_act1 = np.concatenate(input_act1_list[hidx], axis=0)
                        bs, l, dim = cur_input_act1.shape
                        # print(cur_input_act1.shape)
                        input_act1_list[hidx] = cur_input_act1.reshape(bs, l, dim)
                        # out_list[hidx] = out_list[hidx]

                all_level_input_act_list.append(input_act1_list)
                all_out_list.append(out_list)





            sim_func = cosine_similarity
            # sim_func = lambda x, y: np.dot(x, y.T)
            # sim_func = lambda x, y: (x[None, ...] - y[:, None, ...]).sum(axis=-1)
            show_vals = False
            print(f"using function {repr(sim_func).split(' ')[1]}")



            level_corr_mat_accum = []
            # corr_mat = []
            u1, u2, u1_name, u2_name = None, None, None, None

            task_name = 'add_'
            folder_name = f'corr_{task_name}' if not equal_distancing_exp else f'dot_{task_name}'
            folder_name += '_trained' if not model_init else '_init'

            for k in ablation_config:
                if ablation_config[k]:
                    if k == 'func_config':
                        folder_name += f'_{ablation_config[k]}'
                    else: 
                        folder_name += f'_{k}' 
            from vis_utils import get_PE_tendency, generate_tendency_map

            all_stats = None

            get_corr = functools.partial(get_corr_original, 
                                        useful_name_list=useful_name_list,
                                        all_level_input_act_list=all_level_input_act_list,
                                        sim_func=sim_func,
                                        X_n=X_n,
                                        folder_name=folder_name,
                                        fixed_length=fixed_length,
                                        equal_distancing_exp=equal_distancing_exp)
            
            get_corr = functools.partial(get_corr,
                                        level=0,
                                    standard=6,
                                    drop_down=useful_name_list[0],
                                    drop_down2=None,
                                    drop_down3=None,
                                    plot_type='dot',
                                    all_models=False,
                                    before_after='training',
                                    save=True,
                                    abs=False,
                                    save_all=False,
                                    accumulate_all=False,
                                    subplot_layers=True)

            for i in range(5, 100, 5):
                get_corr(sample_idx=i, individual_sample=True)

            get_corr(sample_idx=0, individual_sample=False)
        except:
            pass
            
                    