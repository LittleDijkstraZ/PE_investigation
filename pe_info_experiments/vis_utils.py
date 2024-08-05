
from scipy.stats import spearmanr, pearsonr
import numpy as np
import torch

def levenshteinDistance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

def get_PE_tendency(mat, as_list=False):
    r = 0
    r2 = 0
    r3 = 0
    r4 = 0
    r5 = 0
    r5_counts = 0
    r6 = 0
    r6_count = 0
    mat = mat.detach().cpu().numpy() if isinstance(mat, torch.Tensor) else mat
    for lidx, layer in enumerate(mat[1:]):
        r += (np.diff(layer[:lidx+1], axis=0) > 0).sum()
        r3 += (np.diff(layer[:lidx+1], axis=0) > 0).sum()
        r3 -= (np.diff(layer[:lidx+1], axis=0) <= 0).sum()

        c2p = sum([((layer[j+1:lidx+1]-layer[j]) > 0).sum() for j in range(lidx+1)])
        c2m = sum([((layer[j+1:lidx+1]-layer[j]) <= 0).sum() for j in range(lidx+1)])
        r2 += c2p
        r4 += c2p - c2m

        valid_layer = layer[:lidx+1]
        if len(valid_layer)<3: 
            continue

        # cur_order = valid_layer
        # original_order = valid_layer[np.argsort(valid_layer)]
        # corr = spearmanr(original_order, cur_order).correlation
        # if np.isnan(corr):
            # continue
        # r5 += spearmanr(original_order, cur_order).correlation 


        right_order = np.argsort(valid_layer-np.arange(len(valid_layer))*1e-5)
        right_right_order = np.argsort(right_order)
        original_order = np.arange(len(valid_layer))
        edit_distance = levenshteinDistance(right_order, original_order)
        r5 += edit_distance
        r5_counts += len(valid_layer)
        r_value = pearsonr(original_order, right_right_order)[0]
        if np.isnan(r_value): continue
        r6 += pearsonr(original_order, right_right_order)[0]
        r6_count += 1
        # r5 += pearsonr(original_order, cur_order)[0]

    max_r = np.arange(mat.shape[0]-1).sum()
    t1 = round(r/max_r, 2)

    max_r2 = sum([ np.arange(n+1).sum() for n in np.arange(mat.shape[0]-1)])
    t2 = round(r2/max_r2, 2)

    t3 = round(r3/max_r, 2)

    t4 = round(r4/max_r2, 2)
    
    t5 =  round((r5_counts-r5) / r5_counts, 2)

    t6 = round(r6/r6_count, 2)

    if not as_list:
        return f'({t1},{t2},{t3},{t4},{t5},{t6})'
    else:
        return [t1, t2, t3, t4, t5, t6]

def generate_tendency_map(mat):
    mat = mat.detach().cpu().numpy() if isinstance(mat, torch.Tensor) else mat
    empty_mat = np.zeros_like(mat)
    for lidx, layer in enumerate(mat):
        if lidx == 0:
            continue
        counter = 0
        for j in range(1, lidx+1):
            if layer[j]<=layer[j-1]: # weird but worked ...
                counter = 0
            else:
                counter += 1
            empty_mat[lidx, j] = counter
    
    return empty_mat