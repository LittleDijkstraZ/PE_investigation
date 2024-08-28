import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm
from vis_utils import get_PE_tendency
import os
import gc

def standardize_rows(matrix):
    """Standardize each row of the matrix."""
    mean = matrix.mean(axis=1, keepdims=True)
    std = matrix.std(axis=1, keepdims=True)
    return (matrix - mean) / std

def get_corr(
            useful_name_list,
            all_level_input_act_list,
            sim_func,
            X_n,
            folder_name,
            fixed_length,
            equal_distancing_exp=False,
            sample_idx=0, 
            level=0, 
            standard=6,
            drop_down=[], 
            drop_down2=None, 
            drop_down3=None, 
            plot_type = ['dot', 'corr'],
            all_models = False,
            individual_sample = True,
            before_after = ['training', False,  'attention'],
            save=False,  
            abs=False, 
            save_all=False, 
            accumulate_all=False, 
            subplot_layers=True,
            show_vals=False,):
    # global corr_mat, input_act1_list

    font_path = './timr45w.ttf'  # Update this path
    from matplotlib import font_manager
    # Add the font to Matplotlib's font manager
    font_manager.fontManager.addfont(font_path)
    prop = font_manager.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = prop.get_name()
    plt.rcParams['font.family'] = prop.get_name()
    plt.rcParams.update({'font.size':32})
    plt.rcParams['axes.labelsize'] =32  # Axis labels
    plt.rcParams['xtick.labelsize'] =32  # X-axis tick labels
    plt.rcParams['ytick.labelsize'] =32  # Y-axis tick labels
    plt.rcParams['legend.fontsize'] =32  # Legend
    plt.rcParams['axes.titlesize'] =32  # Title
    # global u1, u2, u1_name, u2_name
    # global all_stats

    idx1 = useful_name_list.index(drop_down)
    idx2 = useful_name_list.index(drop_down2) if drop_down2 is not None else None
    idx3 = useful_name_list.index(drop_down3) if drop_down3 is not None else None

    # model_layers = [l[idx1] for l in all_level_input_act_list if len(l[idx1])!=0]
    # level = min(level, len(model_layers)-1)
    # print(level, idx1, useful_name_list[idx1])

    for l in all_level_input_act_list:
        if l == []:
            return

    # individual_sample = individual_sample if not save_all else False
    if individual_sample :
        if not equal_distancing_exp:
            print(X_n[sample_idx:sample_idx+1])
        else:
            print('eqx')
    save = True if save_all else save
    model_range = range(idx1, idx1+1) if not save_all else range(len(useful_name_list))
    all_stats = dict()
    # for idx1 in tqdm(model_range):
    for idx1 in tqdm(model_range):

        eqn_text = X_n[sample_idx] if not equal_distancing_exp else 'eqx'
        
        models_corr_mat_list = []
        models_cm_tend_list = []
        
        model_name = useful_name_list[idx1]
        acc = model_name.split('_')[0]
        rest = '_'.join(model_name.split('_')[1:])
        try:
            iteration = int(re.search(r'acc_(\d+).pt', model_name).group(1))
        except:
            if 'Init' in model_name:
                iteration = 0
            else:
                iteration = 5000
        imgname = rest.replace('10000_acc_', '').replace('/', '_').replace('.pt', '') + ' ' + eqn_text

        cur_model_act_list1 = [l[idx1] for l in all_level_input_act_list if len(l[idx1])!=0]
        models_to_plot = [cur_model_act_list1]
        
        has_model = True
        if before_after=='training':
            rest = '_'.join(rest.split('.')[0].split('_')[:-1])
            if acc.split('=')[-1]!='0' or iteration!=0:
                print(acc, iteration)
                has_model = False
                continue
            trained_model_name = None
            for m_idx, m_name in enumerate(useful_name_list):
                # if rest in m_name and not (m_name==model_name):
                if rest in m_name and 'trained' in m_name:
                    trained_model_name = m_name
                    # trained_cur_model = model_list[m_idx]
                    m_idx = useful_name_list.index(m_name)
                    cur_model_act_list2 = [l[m_idx] for l in all_level_input_act_list if len(l[m_idx])!=0]
                    break
            assert trained_model_name is not None, 'no trained model found'
            # print(all_out_list[0][idx1][sample_idx])
            models_to_plot.append(cur_model_act_list2)

        elif before_after=='attention':
            models_to_plot.append(cur_model_act_list1)

        if drop_down2 is not None:
            cur_model_act_list2 = [l[idx2] for l in all_level_input_act_list if len(l[idx2])!=0]
            models_to_plot.append(cur_model_act_list2)
        if drop_down3 is not None:
            cur_model_act_list3 = [l[idx3] for l in all_level_input_act_list if len(l[idx3])!=0]
            models_to_plot.append(cur_model_act_list3)
        

        for cur_model_act_list in models_to_plot:
            corr_mat_list = []
            level_cm_tend_list = []
            for level in tqdm(range(len(cur_model_act_list))):
                input_act1_list = cur_model_act_list[level]
                # global level_corr_mat_accum

                u1, u2 = cur_model_act_list[level], cur_model_act_list[level]
                # u1 = u1.T
                # u2 = u2.T
                if individual_sample:
                    u1, u2 = u1[sample_idx:sample_idx+1], u2[sample_idx:sample_idx+1]
            
                if plot_type=='corr':
                    corr_mat = []
                    for b1, b2 in zip(u1, u2):
                        u1_standardized = standardize_rows(u1)
                        u2_standardized = standardize_rows(u2)
                        corr_mat.append(np.dot(u1_standardized, u2_standardized.T) / (u1.shape[1]))
                elif plot_type=='dot':
                    corr_mat = []
                    for b1, b2 in zip(u1, u2):
                        corr_mat.append(sim_func(b1, b2))
                    corr_mat_all = np.array(corr_mat)
                    corr_mat = corr_mat_all.mean(axis=0) 

                if abs:
                    corr_mat = np.abs(corr_mat)
                else:
                    pass

                
                cm_tend_list = []
                for _sidx in range(len(corr_mat_all)):
                    cm_tend_list.append(get_PE_tendency(corr_mat_all[_sidx], as_list=True)[standard])
                        
                # if accumulate_all:
                #     level_corr_mat_accum.append(corr_mat[None, ...])

                
                corr_mat_list.append(corr_mat)
                level_cm_tend_list.append(cm_tend_list)               


                if not subplot_layers:
                    vec_dim = corr_mat.shape[0]//fixed_length
                    total_sum = np.abs(corr_mat).sum()
                    block_sum = 0
                    for i in range(0, len(corr_mat), vec_dim):
                        block_sum += np.abs(corr_mat[i:i+vec_dim, i:i+vec_dim]).sum()
                    ratio = block_sum/(total_sum-block_sum)

                    plt.figure(figsize=(6, 5), dpi=120)
                    plt.imshow(corr_mat, cmap='Reds') #, interpolation='nearest')

                    extra_text = 'Absolute ' if abs else ''
                    plt.colorbar(label=f'{extra_text}Correlation Coefficient')

                    os.makedirs(f'./saved_plots_{folder_name}/', exist_ok=True)
                    plt.savefig(f'./saved_plots_{folder_name}/{useful_name_list[idx1]}_avg_{abs}.pdf')
                    plt.close()
            models_corr_mat_list.append(corr_mat_list)

            models_cm_tend_list.append(level_cm_tend_list)
        # corr_mat_list = models_corr_mat_list
        
        if subplot_layers and has_model:
            handel=f'{before_after}_{standard}_indi={individual_sample}/'
            folder_dir = f'./saved_plots_sim/{handel}'
            # if save:
            #     if f'{folder_dir}/{imgname}.pdf' in glob.glob(f'{folder_dir}/{imgname}.pdf'):
            #         continue
            fig, axs = plt.subplots(len(models_corr_mat_list), len(models_corr_mat_list[0]), figsize=(6*len(models_corr_mat_list[0]), 5.6*len(models_corr_mat_list)))  # 6 subplots in a row, adjust size as needed
            
            for level in range(len(models_corr_mat_list[0])):

                for cm_idx, corr_mat_list in enumerate(models_corr_mat_list):
                    try:
                        corr_mat = corr_mat_list[level]
                    except:
                        print(corr_mat_list)
                        raise ValueError

                    # vec_dim = n_embd
                    cm_tend = get_PE_tendency(corr_mat, as_list=True)[standard]

                    vec_dim = corr_mat.shape[0]//fixed_length

                    total_sum = np.abs(corr_mat).sum()
                    block_sum = 0
                    for i in range(0, len(corr_mat), vec_dim):
                        block_sum += np.abs(corr_mat[i:i+vec_dim, i:i+vec_dim]).sum()
                    ratio = block_sum/(total_sum-block_sum)

                    axloc = axs[level] if len(models_corr_mat_list)==1 else axs[cm_idx, level]

                    cm = axloc.imshow(corr_mat, cmap='Reds') #, interpolation='nearest')
                    # show the number on each block
                    if show_vals:
                        for i in range(0, len(corr_mat), vec_dim):
                            for j in range(0, len(corr_mat), vec_dim):
                                text = axloc.text(j, i, f'{corr_mat[i:i+vec_dim, j:j+vec_dim].sum():.02f}',
                                                    ha="center", va="center", color="black", fontsize=6)

                    training_status = ''
                    if before_after=='training':
                        training_status = 'Trained ' if cm_idx==1 else 'Init '
                    title_text = f'{training_status}Layer {level} ({cm_tend})' if level > 0 \
                        else f'Embeddings ({cm_tend})'                  
                    axloc.set_title(title_text)
                    cbar = plt.colorbar(cm, ax=axloc, orientation='vertical', fraction=0.05, )
                    # Get the scalar formatter from the colorbar
                    from matplotlib.ticker import FormatStrFormatter, MaxNLocator
                    formatter = FormatStrFormatter('%.1f')
                    axloc.yaxis.set_major_formatter(FormatStrFormatter('%d'))
                    axloc.yaxis.set_major_locator(MaxNLocator(nbins=5))
                    axloc.xaxis.set_major_locator(MaxNLocator(nbins=5))

                    # Set the desired format (e.g., rounding to 2 decimal places)
                    # formatter.set_useOffset(False)  # Disable offset if necessary

                    # Apply the formatter to the colorbar
                    cbar.formatter = formatter
                    cbar.update_ticks()
            
            plt.tight_layout()
            fig.subplots_adjust(left=0.03, right=0.975, top=0.95, bottom=0.06, hspace=0.25)
            if save:
                os.makedirs(f'{folder_dir}', exist_ok=True)

                plt.savefig(f'{folder_dir}/{imgname} sample.pdf', format='pdf')
                plt.close()
                gc.collect()
            else:
                plt.show()
            
            '''Plotting the Distribution'''
            
            fig, axs = plt.subplots(len(models_corr_mat_list), len(models_corr_mat_list[0]), figsize=(6*len(models_corr_mat_list[0]), 5.6*len(models_corr_mat_list)))  # 6 subplots in a row, adjust size as needed
            levels_stats = []
            for level in range(len(models_corr_mat_list[0])):
                for cm_idx, level_cm_tend_list in enumerate(models_cm_tend_list):
                    axloc = axs[level] if len(models_corr_mat_list)==1 else axs[cm_idx, level]
                    axloc.hist(level_cm_tend_list[level], bins=32)
                    training_status = ''
                    if before_after=='training':
                        training_status = 'Trained ' if cm_idx==1 else 'Init '
                    mean_tend = round(np.mean(level_cm_tend_list[level]), 2)
                    std_tend = round(np.std(level_cm_tend_list[level]), 2)
                    levels_stats.append(mean_tend)
                    title_text = f'{training_status}Layer {level} ({mean_tend}, {std_tend})' if level > 0 \
                        else f'Embeddings ({mean_tend}, {std_tend})'                
                    axloc.set_title(title_text)                    
                    axloc.set_xlim(-1.05, 1.05)

            
            plt.tight_layout()
            fig.subplots_adjust(left=0.03, right=0.975, top=0.95, bottom=0.06, hspace=0.25)
            dist_name = f'{folder_dir}/{imgname} dist.pdf'
            if save:
                os.makedirs(f'{folder_dir}', exist_ok=True)
                plt.savefig(dist_name, format='pdf')
                plt.close()
                gc.collect()
            else:
                plt.show()
            
            all_stats[dist_name]=levels_stats

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
        return all_stats