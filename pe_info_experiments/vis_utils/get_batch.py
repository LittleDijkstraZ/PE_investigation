import numpy as np
import torch



# non_causal_fix_length = 15
# non_causal_fix_length = 27
answer_length = 1

def get_batch_original(
        train_data,
        val_data,
        split, 
        data_encoder,
        
        autoregressive_training=False, 
        batch_size=256,
        causal_training=True,
        train_both=False,
        train_data2=None,
        val_data2=None,
        data_ratio=0.2,
        block_size=1024,
        device='cuda',):

        space_token = data_encoder(' ')[0]
        switch_line_token = data_encoder('\n')[0]
        equal_token = data_encoder('=')[0]
        dollar_token = data_encoder('$')[0]
        
        attn_mask = None
        w = None
        data = train_data if split == 'train' else val_data
        if train_both:
            data2 = train_data2 if split == 'train' else val_data2
            batch_size2 = int(batch_size*data_ratio)
            ix = torch.randint(len(data) - block_size, (batch_size-batch_size2,))
            ix2 = torch.randint(len(data2) - block_size, (batch_size2,))
        else:
            if causal_training:
                ix = torch.randint(len(data) - block_size, (batch_size,))
            else:
                split_points = np.where(data == (switch_line_token))[0]
                answer_split_points = np.where(data == (equal_token))[0]
                answer_length_list = split_points - answer_split_points - 1
                split_points = split_points + 1  # i should have had this
                split_points = np.hstack([np.array([0]), split_points.flatten()])

                sample_length_list = np.diff(split_points)
                start_points = split_points[:-1]

                randidx = np.random.permutation(len(start_points))[:batch_size]
                ix = start_points[randidx]
                sample_length_list = sample_length_list[randidx]
                answer_length_list = answer_length_list[randidx]

        if causal_training:
            x = torch.stack(
                [torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
            y = torch.stack(
                [torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
        else:
            remove_dollar_count = 1 if dollar_token in data else 0
            if not autoregressive_training:
                cur_answer_length_list = np.random.randint(
                    1+remove_dollar_count, answer_length_list+1) + 1
            else:
                cur_answer_length_list = answer_length_list + 1 + 4
            x = [data[ix[i]:ix[i]+sample_length_list[i]-cur_answer_length_list[i]
                    ].astype(np.int64) for i in range(len(ix))]
            x_len = [len(x[i]) for i in range(len(x))]
            pad_to_length = max(x_len)
            min_length = min(x_len)
            # only do padding when the length is not equal
            if pad_to_length > min_length:
                x = np.vstack([np.pad(x[i], (pad_to_length-len(x[i]), 0), mode='constant',
                            constant_values=space_token) for i in range(len(x))])
                attn_mask = np.ones_like(x)
                # mask out the paddings
                attn_mask[x == space_token] = 0
                attn_mask = attn_mask[..., None]
                attn_mask = attn_mask @ attn_mask.transpose(0, 2, 1)
                attn_mask = attn_mask.astype(bool)
                if (attn_mask == 1).all():
                    attn_mask = None
                else:
                    attn_mask = torch.from_numpy(attn_mask)
            else:
                x = np.vstack(x)
                attn_mask = None

            x = torch.from_numpy(x)
            # predict the next digit
            if not autoregressive_training:
                y = torch.stack([torch.from_numpy((data[ix[i]+sample_length_list[i]-cur_answer_length_list[i]:ix[i] +
                                1+sample_length_list[i]-cur_answer_length_list[i]]).astype(np.int64)) for i in range(len(ix))])
            else:
                y = [torch.from_numpy((data[ix[i]+sample_length_list[i]-cur_answer_length_list[i]:ix[i]-1+sample_length_list[i]]).astype(np.int64)) for i in range(len(ix))]
                max_len_y = max([len(y[i]) for i in range(len(y))])
                y = np.vstack([np.pad(y[i], (0, max_len_y-len(y[i])), mode='constant',
                            constant_values=space_token) for i in range(len(y))])
                y = torch.from_numpy(y)
                w = torch.ones_like(y)
                w[y == space_token] = 0

        if train_both:
            x2 = torch.stack(
                [torch.from_numpy((data2[i:i+block_size]).astype(np.int64)) for i in ix2])
            y2 = torch.stack(
                [torch.from_numpy((data2[i+1:i+1+block_size]).astype(np.int64)) for i in ix2])
            x = torch.cat([x, x2])
            y = torch.cat([y, y2])

        if device == 'cuda':
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(
                device, non_blocking=True)
            if autoregressive_training:
                w = w.pin_memory().to(device, non_blocking=True)
            if attn_mask is not None:
                attn_mask = attn_mask.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)

            if attn_mask is not None:
                attn_mask = attn_mask.to(device)

        # attn_mask = None
        return x, y, attn_mask, w
