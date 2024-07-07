import ipywidgets as widgets
import gc
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import cosine_similarity

font_path = './timr45w.ttf'  # Update this path
from matplotlib import font_manager
# Add the font to Matplotlib's font manager
font_manager.fontManager.addfont(font_path)
prop = font_manager.FontProperties(fname=font_path)
plt.rcParams['font.family'] = prop.get_name()
plt.rcParams['font.family'] = prop.get_name()
plt.rcParams.update({'font.size':22})
plt.rcParams['axes.labelsize'] =22  # Axis labels
plt.rcParams['xtick.labelsize'] =22  # X-axis tick labels
plt.rcParams['ytick.labelsize'] =22  # Y-axis tick labels
plt.rcParams['legend.fontsize'] =22  # Legend
plt.rcParams['axes.titlesize'] =22  # Title


sim_func = cosine_similarity
# sim_func = lambda x, y: np.dot(x, y.T)
# sim_func = lambda x, y: (x[None, ...] - y[:, None, ...]).sum(axis=-1)
show_vals = False
print(f"using function {repr(sim_func).split(' ')[1]}")

def standardize_rows(matrix):
    """Standardize each row of the matrix."""
    mean = matrix.mean(axis=1, keepdims=True)
    std = matrix.std(axis=1, keepdims=True)
    return (matrix - mean) / std

level_corr_mat_accum = []
corr_mat = None
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

def get_corr(idx1, 
             level=0, 
             drop_down=[], 
             plot_type = ['dot', 'corr'],
             individual = True,
             save=False,  
             abs=True, 
             save_all=False, 
             accumulate_all=False, 
             subplot_layers=True):
    global corr_mat, input_act1_list
    global u1, u2, u1_name, u2_name
    rand_state = 'init_' if model_init else ''

    if drop_down != []:
        idx1 = useful_name_list.index(drop_down)

    model_layers = [l[idx1] for l in all_level_input_act_list if len(l[idx1])!=0]
    level = min(level, len(model_layers)-1)
    print(level, idx1, useful_name_list[idx1])

    if not (save_all or accumulate_all):
        if not subplot_layers:
            idx2 = idx1
            input_act1_list = all_level_input_act_list[level]
            u1, u2 = input_act1_list[idx1], input_act1_list[idx2]

            u1 = u1.T
            u2 = u2.T
            
            # print(useful_name_list[idx1][0].split('sd')[-1], useful_name_list[idx2][0].split('sd')[-1])

            print(u2.shape)

            if u2.shape[1] != 1:
                # Standardize each row of u1 and u2
                u1_standardized = standardize_rows(u1)
                u2_standardized = standardize_rows(u2)

                corr_mat = np.dot(u1_standardized, u2_standardized.T) / (u1.shape[1] )
            else:
                u1 = u1.reshape(-1, n_embd)
                u2 = u2.reshape(-1, n_embd)
                # import cosine similarity
                corr_mat = sim_func(u1, u2)

                # corr_mat = np.dot(u1, u2.T)

            if abs:
                corr_mat = np.abs(corr_mat)
            else:
                pass
            vec_dim = corr_mat.shape[0]//fixed_length
            total_sum = np.abs(corr_mat).sum()
            block_sum = 0
            for i in range(0, len(corr_mat), vec_dim):
                block_sum += np.abs(corr_mat[i:i+vec_dim, i:i+vec_dim]).sum()
            ratio = block_sum/(total_sum-block_sum)

            # Assuming you have calculated 'corr_mat' as described in the previous answer

            # Create a heatmap of corr_mat
            plt.figure(figsize=(6, 5), dpi=120)
            # plt.imshow(corr_mat, cmap='seismic') #, interpolation='nearest')
            plt.imshow(corr_mat, cmap='Reds') #, interpolation='nearest')

            extra_text = 'Absolute ' if abs else ''
            plt.colorbar(label=f'{extra_text}Correlation Coefficient', fraction=0.06, pad=0.04,)

            # Show the plot
            
            nope1 = 'nope_' if 'nope' in useful_name_list[idx1][1] else ''
            
            u1_name = '_'.join(useful_name_list[idx1][1].split('_')[2:])
            u1_name = nope1 + u1_name.split('_')[-1] + '_' + '_'.join(u1_name.split('_')[:-1])
            
            nope2 = 'nope_' if 'nope' in useful_name_list[idx1][1] else ''
            
            u2_name = '_'.join(useful_name_list[idx2][1].split('_')[2:])
            u2_name = nope2+u2_name.split('_')[-1] + '_' +'_'.join(u2_name.split('_')[:-1])

            extra_self = 'Self' if u1_name == u2_name else ''
            # plt.title(f'{extra_self} Correlation Matrix for  {ratio:.2f}')
            # print(u1_name, u2_name)
            if save:
                os.makedirs(f'./saved_plots_{folder_name}/', exist_ok=True)
                plt.savefig(f'./saved_plots_{folder_name}/{task_name+folder_name}_{rand_state}_{u2_name}_layer{level}_{ratio:.03f}_{abs}.pdf')
            
            # close img
            # plt.close()
            plt.show()
        else:
            fig, axs = plt.subplots(1, len(model_layers), figsize=(36, 5))  # 6 subplots in a row, adjust size as needed

            global_min, global_max = float('inf'), float('-inf')
            corr_mat_list= [ ]
            for level in range(len(model_layers)):
                idx2 = idx1
                # input_act1_list = model_layers[level]
                u1, u2 = model_layers[level], model_layers[level]

                u1 = u1.T
                u2 = u2.T

                if plot_type=='corr':
                    u1_standardized = standardize_rows(u1)
                    u2_standardized = standardize_rows(u2)
                    corr_mat = np.dot(u1_standardized, u2_standardized.T) / (u1.shape[1])
                elif  plot_type=='dot':
                    corr_mat = []
                    for b1, b2 in zip(u1, u2):
                    # corr_mat = np.dot(u1, u2.T)
                        corr_mat.append(sim_func(b1, b2))
                    corr_mat = np.array(corr_mat)
                    corr_mat = corr_mat.mean(axis=0) 

                if abs:
                    corr_mat = np.abs(corr_mat)
                    
                corr_mat_list.append(corr_mat)
                
                global_min = min(global_min, corr_mat.min())
                global_max = max(global_max, corr_mat.max())


            for level in range(len(model_layers)):
                
                corr_mat = corr_mat_list[level]

                # vec_dim = n_embd
                vec_dim = corr_mat.shape[0]//fixed_length

                total_sum = np.abs(corr_mat).sum()
                block_sum = 0
                for i in range(0, len(corr_mat), vec_dim):
                    block_sum += np.abs(corr_mat[i:i+vec_dim, i:i+vec_dim]).sum()
                ratio = block_sum/(total_sum-block_sum)

                cm = axs[level].imshow(corr_mat, cmap='Reds') #, interpolation='nearest')
                
                # show the number on each block
                if show_vals:
                    for i in range(0, len(corr_mat), vec_dim):
                        for j in range(0, len(corr_mat), vec_dim):
                            text = axs[level].text(j, i, f'{corr_mat[i:i+vec_dim, j:j+vec_dim].sum():.02f}',
                                                ha="center", va="center", color="black", fontsize=6)
                axs[level].set_title(f'Layer {level}')
                plt.colorbar(cm, ax=axs[level], orientation='vertical', fraction=0.06, pad=0.04,)

            extra_text = 'Absolute ' if abs else ''

            
            nope1 = 'nope_' if 'nope' in useful_name_list[idx1][1] else ''
            
            u1_name = '_'.join(useful_name_list[idx1][1].split('_')[2:])
            u1_name = nope1 + u1_name.split('_')[-1] + '_' + '_'.join(u1_name.split('_')[:-1])
            
            nope2 = 'nope_' if 'nope' in useful_name_list[idx1][1] else ''
            
            u2_name = '_'.join(useful_name_list[idx2][1].split('_')[2:])
            u2_name = nope2+u2_name.split('_')[-1] + '_' +'_'.join(u2_name.split('_')[:-1])

            extra_self = 'Self' if u1_name == u2_name else ''
            # plt.title(f'{extra_self} Correlation Matrix for  {ratio:.2f}')
            # print(u1_name, u2_name)
            if save:
                os.makedirs(f'./saved_plots_{folder_name}/', exist_ok=True)
                plt.savefig(f'./saved_plots_{folder_name}/{task_name+folder_name}_{len(all_level_input_act_list)}layers_{rand_state}_{u2_name}_{ratio:.03f}_{abs}.pdf')
            
            # close img
            # plt.close()
            plt.show()

    else:
        for l in all_level_input_act_list:
            if l == []:
                return

        for idx1 in tqdm(range(len(input_act1_list))):
            corr_mat_list = []

            for level in range(len(all_level_input_act_list)):
                input_act1_list = all_level_input_act_list[level]
                global level_corr_mat_accum

                idx2 = idx1
                u1, u2 = input_act1_list[idx1], input_act1_list[idx2]
                u1 = u1.T
                u2 = u2.T
                
                if u2.shape[1] != 1:
                    u1_standardized = standardize_rows(u1)
                    u2_standardized = standardize_rows(u2)
                    corr_mat = np.dot(u1_standardized, u2_standardized.T) / (u1.shape[1] )
                else:
                    u1 = u1.reshape(-1, n_embd)
                    u2 = u2.reshape(-1, n_embd)
                    corr_mat = sim_func(u1, u2) 

                if abs:
                    corr_mat = np.abs(corr_mat)
                else:
                    pass
            
                if accumulate_all:
                    level_corr_mat_accum.append(corr_mat[None, ...])
                if not save_all:
                    continue 
                
                corr_mat_list.append(corr_mat)

                if not subplot_layers:
                    vec_dim = corr_mat.shape[0]//fixed_length
                    total_sum = np.abs(corr_mat).sum()
                    block_sum = 0
                    for i in range(0, len(corr_mat), vec_dim):
                        block_sum += np.abs(corr_mat[i:i+vec_dim, i:i+vec_dim]).sum()
                    ratio = block_sum/(total_sum-block_sum)

                    # Assuming you have calculated 'corr_mat' as described in the previous answer

                    # Create a heatmap of corr_mat
                    plt.figure(figsize=(6, 5), dpi=120)
                    # plt.imshow(corr_mat, cmap='seismic') #, interpolation='nearest')
                    plt.imshow(corr_mat, cmap='Reds') #, interpolation='nearest')

                    extra_text = 'Absolute ' if abs else ''
                    plt.colorbar(label=f'{extra_text}Correlation Coefficient')

                    # Set axis labels and title
                    # plt.xlabel('U2 Entries')
                    # plt.ylabel('U1 Entries')

                    # Show the plot
                    
                    nope1 = 'nope_' if 'nope' in useful_name_list[idx1][1] else ''
                    
                    u1_name = '_'.join(useful_name_list[idx1][1].split('_')[2:])
                    u1_name = nope1 + u1_name.split('_')[-1] + '_' + '_'.join(u1_name.split('_')[:-1])
                    
                    nope2 = 'nope_' if 'nope' in useful_name_list[idx1][1] else ''
                    
                    u2_name = '_'.join(useful_name_list[idx2][1].split('_')[2:])
                    u2_name = nope2+u2_name.split('_')[-1] + '_' +'_'.join(u2_name.split('_')[:-1])

                    extra_self = 'Self' if u1_name == u2_name else ''
                    # plt.title(f'{extra_self} Correlation Matrix for  {ratio:.2f}')
                    # print(ratio)
                    # # print(u1_name, u2_name)
                    os.makedirs(f'./saved_plots_{folder_name}/', exist_ok=True)
                    # plt.savefig(f'./saved_plots_{folder_name}/{task_name+folder_name}_{rand_state}_{u2_name}_layer{level}_{ratio:.03f}_{abs}.pdf')
                    plt.savefig(f'./saved_plots_{folder_name}/{useful_name_list[idx1]}_avg_{abs}.pdf')
                    
                    # close img
                    plt.close()

            if subplot_layers:
                acc = useful_name_list[idx1].split('_')[0]
                rest = '_'.join(useful_name_list[idx1].split('_')[1:])
                imgname = rest.replace('10000_acc_', '').replace('/', '_').replace('.pt', '') + '_' + acc
                if f'./saved_plots_{folder_name}/{imgname}.pdf' in glob.glob(f'./saved_plots_{folder_name}/*.pdf'):
                    continue
                fig, axs = plt.subplots(1, len(all_level_input_act_list), figsize=(40, 5))  # 6 subplots in a row, adjust size as needed
                for level in range(len(all_level_input_act_list)):
                    
                    corr_mat = corr_mat_list[level]

                    # vec_dim = n_embd
                    vec_dim = corr_mat.shape[0]//fixed_length

                    total_sum = np.abs(corr_mat).sum()
                    block_sum = 0
                    for i in range(0, len(corr_mat), vec_dim):
                        block_sum += np.abs(corr_mat[i:i+vec_dim, i:i+vec_dim]).sum()
                    ratio = block_sum/(total_sum-block_sum)

                    cm = axs[level].imshow(corr_mat, cmap='Reds') #, interpolation='nearest')
                    # show the number on each block
                    if show_vals:
                        for i in range(0, len(corr_mat), vec_dim):
                            for j in range(0, len(corr_mat), vec_dim):
                                text = axs[level].text(j, i, f'{corr_mat[i:i+vec_dim, j:j+vec_dim].sum():.02f}',
                                                    ha="center", va="center", color="black", fontsize=6)

                    
                    title_text = f'Layer {level}' if level > 0 else 'Token Embeddings'                
                    axs[level].set_title(title_text)
                    plt.colorbar(cm, ax=axs[level], orientation='vertical', fraction=0.06, pad=0.04,)

                os.makedirs(f'./saved_plots_{folder_name}/', exist_ok=True)
                

                plt.savefig(f'./saved_plots_{folder_name}/{imgname}.pdf', format='pdf')
                plt.close()
                gc.collect()


            # if idx1 >=10:
            #     raise ValueError
            if accumulate_all:
                level_corr_mat_accum = np.vstack(level_corr_mat_accum)
                print(level_corr_mat_accum.shape)
                level_corr_mat_accum = level_corr_mat_accum.mean(axis=0)
                # Create a heatmap of corr_mat
                if not subplot_layers:
                    plt.figure(figsize=(6, 5), dpi=120)
                    # plt.imshow(corr_mat, cmap='seismic') #, interpolation='nearest')
                    plt.imshow(level_corr_mat_accum, cmap='Reds') #, interpolation='nearest')

                    extra_text = 'Absolute ' if abs else ''
                    plt.colorbar(label=f'{extra_text}Correlation Coefficient')
                    plt.show()
                corr_mat  =  level_corr_mat_accum
                level_corr_mat_accum = []
            
           


# all_drop_down = [e[1] for e in useful_name_list]

widgets.interact(get_corr, idx1=(0, len(input_act1_list)-1), drop_down=useful_name_list, level=(0,12))


# idxes = [
# # [0,0], [0,1], [0,2], [0,3], [0,4],
# # [1,1], [2,2], [3,3], [4,4],
# # [4,1], [4,2], [4,3], 
# [1,2], [1,3],c
# ]
# from tqdm.auto import tqdm
# for idx1, idx2 in tqdm(idxes):
#     get_corr(idx1, idx2, save=True)
    
# all residual vs picking ones that are illustrative