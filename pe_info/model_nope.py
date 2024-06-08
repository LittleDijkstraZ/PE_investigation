"""
File based on the original NanoGPT codebase.
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
import dataclasses
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

class RelativePositionBias(nn.Module):
    def __init__(self, bidirectional=False, num_buckets=32, max_distance=128, n_heads=2):
        super(RelativePositionBias, self).__init__()
        self.bidirectional = bidirectional
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.n_heads = n_heads
        self.relative_attention_bias = nn.Embedding(self.num_buckets, self.n_heads)

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on

        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer

        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
        """
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        # now relative_position is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_postion_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_postion_if_large = torch.min(
            relative_postion_if_large, torch.full_like(relative_postion_if_large, num_buckets - 1)
        )

        relative_buckets += torch.where(is_small, relative_position, relative_postion_if_large)
        return relative_buckets

    def compute_bias(self, query_length, key_length):
        """Compute binned relative position bias"""
        context_position = torch.arange(
            query_length, dtype=torch.long, device=self.relative_attention_bias.weight.device
        )[:, None]
        memory_position = torch.arange(
            key_length, dtype=torch.long, device=self.relative_attention_bias.weight.device
        )[None, :]
        relative_position = memory_position - context_position  # shape (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=self.bidirectional,
            num_buckets=self.num_buckets,
        )
        values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        return values

    def forward(self, qlen, klen):
        return self.compute_bias(qlen, klen)  # shape (1, num_heads, qlen, klen)

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(SinusoidalPositionalEncoding, self).__init__()
        # Create a long enough 'pe' matrix with positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        # Register as a buffer in PyTorch (not a parameter, but should be part of the state)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, sequence_length, embedding_dim]
        """
        # Add positional encoding to input tensor 'x'
        x = x + self.pe[:, :x.size(1), :]
        return x

# @torch.jit.script # good to enable when not using torch.compile, disable when using (our default)
def new_gelu(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


from rotary_embedding_torch import RotaryEmbedding

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        if config.layer_pe == 'rotary':
            self.rotary_emb = RotaryEmbedding(config.n_embd // config.n_head,
                                              freqs_for='lang')
            print('rotary in use')
        elif config.layer_pe == 'T5':
            self.relative_emb = RelativePositionBias(max_distance=config.block_size ,n_heads=config.n_head)
            print('T5 in use')

        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)

        self.identity = nn.Identity() # this is only for helping to take out the results after attention through forward hook

        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)


        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.use_flash = config.use_flash
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and self.use_flash
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))
        else:
            print("Using Flash Attention")
        
        self.causal = bool(not config.not_causal)
        self._reset_parameters() 
        self.config = config

    def forward(self, x, attn_mask=None, ablation_config=dict()):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        if ablation_config.get('shrink_x', False):
            shrink_factor = float(ablation_config['shrink_x'])
            q, k, v = x/shrink_factor, x/shrink_factor, x/shrink_factor
        else:
            q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
            if ablation_config.get('no_Q', False):
                q = x
            if ablation_config.get('no_K', False):
                k = x
            if ablation_config.get('no_V', False):
                v = x
        if ablation_config.get('V_out', False):
            if ablation_config.get('no_V', False):
                print('V_out is set but no_V is set, this is not possible')
            return x

        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        if self.config.layer_pe == 'rotary':
            q = self.rotary_emb.rotate_queries_or_keys(q)
            k = self.rotary_emb.rotate_queries_or_keys(k)
        elif self.config.layer_pe == 'T5':
            if attn_mask is None:
                attn_mask = self.relative_emb(T, T) 
            else:
                attn_mask = attn_mask + self.relative_emb(T, T)
            
        if attn_mask is not None:
            attn_mask = torch.repeat_interleave(attn_mask.unsqueeze(1), self.n_head, dim=1) # (B, nh, T, T)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        func_config = ablation_config.get('func_config', False)
        if self.flash and not func_config:
            # efficient attention using Flash Attention CUDA kernels
            if self.causal:
                if attn_mask is None:
                    y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
                else: # since torch doeds not allow setting both attn_mask and is_causal, we have to do it manually
                    temp_mask = torch.ones(1, 1, T, T, dtype=torch.bool).tril(diagonal=0).to(attn_mask.device)
                    # diag_mask = torch.diag_embed(torch.ones(T, dtype=torch.bool)).unsqueeze(0).unsqueeze(0).to(attn_mask.device)
                    if attn_mask.dtype == torch.bool:
                        attn_mask = attn_mask & temp_mask
                        # attn_mask = attn_mask | diag_mask
                    else:
                        attn_mask = attn_mask.masked_fill(temp_mask == 0, float('-inf'))
                        attn_mask = attn_mask[0,0]
                        # attn_mask = attn_mask.masked_fill(diag_mask == 0, float('-inf'))
                    y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0)

            else:
                if attn_mask is None:
                    y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0)
                else:
                    diag_mask = torch.diag_embed(torch.ones(T, dtype=torch.bool)).unsqueeze(0).unsqueeze(0).to(attn_mask.device)
                    attn_mask = attn_mask | diag_mask
                    
                    y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            if not func_config:
                func_config = 'softmax' # go to default
            if self.causal:
                bias =  torch.tril(torch.ones(T, T)).view(1, 1, T, T).to(att.device)
                if func_config=='softmax':
                    att = att.masked_fill(bias[:,:,:T,:T] == 0, float('-inf'))
                else:
                    att = att.masked_fill(bias[:,:,:T,:T] == 0, 0)

            if func_config=='softmax':
                att = F.softmax(att, dim=-1)
            elif func_config=='abs':
                att = att.abs()
                attn_sum = att.abs().sum(dim=-1, keepdim=True)
                att = att / attn_sum
            elif func_config=='relu':
                att = F.relu(att)
                attn_sum = att.sum(dim=-1, keepdim=True)
                att = att / attn_sum
            elif func_config=='gelu':
                att = new_gelu(att)
                attn_sum = att.sum(dim=-1, keepdim=True)
                att = att / attn_sum
            elif func_config=='gelu_divabs':
                att = new_gelu(att)
                attn_sum = att.abs().sum(dim=-1, keepdim=True)
                att = att / attn_sum
            elif func_config=='sigmoid':
                att = torch.sigmoid(att)
                attn_sum = att.sum(dim=-1, keepdim=True)
                att = att / attn_sum
            elif func_config=='divsum':
                attn_sum = att.sum(dim=-1, keepdim=True)
                att = att / attn_sum
            elif func_config=='nodiv':
                att = att
            elif func_config=='same':
                print('using same')
                att = torch.ones_like(att).to(att.device)
                att = att / att.sum(dim=-1, keepdim=True)


                
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.identity(y)
        y = self.resid_dropout(self.c_proj(y))
        return y
    

    def _reset_parameters(self, std=0.02):
        
        ## random normal
        # nn.init.normal_(self.c_attn.weight, mean=0, std=std) 
        # nn.init.normal_(self.c_proj.weight, mean=0, std=std)

        ## xavier
        # print('xavier init')
        # nn.init.xavier_uniform_(self.c_attn.weight)
        # nn.init.xavier_uniform_(self.c_proj.weight)

        ## kaiming
        # print('kaiming init')
        # nn.init.kaiming_uniform_(self.c_attn.weight, a=math.sqrt(5))
        # nn.init.kaiming_uniform_(self.c_proj.weight, a=math.sqrt(5))

        ## orthogonal
        # print('orthogonal init')
        # nn.init.orthogonal_(self.c_attn.weight)
        # nn.init.orthogonal_(self.c_proj.weight)

        ## uniform
        # print('uniform init')
        # nn.init.uniform_(self.c_attn.weight, 0.5, 0.6)
        # nn.init.uniform_(self.c_proj.weight, 0.5, 0.6)

        ## constant
        # print('constant init')
        # nn.init.constant_(self.c_attn.weight, 0.5)
        # print(self.c_attn.weight)
        # nn.init.constant_(self.c_proj.weight, 0.5)

        return 


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = new_gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
    
    def _reset_parameters(self):
        nn.init.normal_(self.c_fc.weight)
        nn.init.normal_(self.c_proj.weight)
        return 

from functools import partial

class Block(nn.Module):
    # create a network
    # verify by permutation
    # theoretically casual shall not make differnce

    def __init__(self, config, id):
        super().__init__()
        self.config = dataclasses.replace(config)
        self.config.not_causal = config.not_causal[id]
        self.layerwise_pe = config.layerwise_pe[id]

        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        
        self.config.layer_pe = config.layer_pe if self.layerwise_pe else None
        self.attn = CausalSelfAttention(self.config)

        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)
        self.id = id
        # jason's 

        self.use_residual = config.use_residual[id]
        self.no_att_residual = config.no_att_residual[id] if self.use_residual else True
        self.no_mlp_residual = config.no_mlp_residual[id] if self.use_residual else True
        self.pe_type = config.layer_pe
        self.permute = config.permute[id]
        self.permute_length = config.permute_length

        self.permute_func = partial(self.permute_func_base, permute_type=self.permute, permute_length=self.permute_length)

        if self.permute:
            # randx = torch.nn.Parameter(torch.randperm(config.block_size).to(torch.int64), requires_grad=False)
            # self.randx = [torch.nn.Parameter(torch.randperm(i).to(torch.long), requires_grad=False) for i in range(config.block_size+1)]
            if self.permute == 'shuffle':
                perm_idx = torch.hstack([torch.randperm(self.permute_length), torch.arange(self.permute_length, config.block_size+1)]).to(torch.long)
                self.register_buffer('randx', perm_idx)
            if self.permute == 'remove':
                voids = torch.zeros(1, self.permute_length, self.config.n_embd)
                self.register_buffer('voids', voids)
                
            self.res1_weight = nn.Parameter(torch.ones(1), requires_grad=True)
            self.res2_weight = nn.Parameter(torch.ones(1), requires_grad=True)
            self.res_weights = [self.res1_weight, self.res2_weight]



        if self.layerwise_pe:
            self.block_size = config.block_size
            if config.layer_pe == 'original':
                self.layer_wpe = nn.Embedding(self.block_size, config.n_embd)
            elif config.layer_pe == 'sin':
                self.layer_wpe = SinusoidalPositionalEncoding(config.n_embd, self.block_size)
          

        print(f"Block {id}: {self.use_residual} | att_res {not self.no_att_residual} | perm {self.permute} | mlp_res {not self.no_mlp_residual} | layerwise_pe {self.layerwise_pe} | casual {not self.config.not_causal}")

    def permute_func_base(self, x, res_weight_id, permute_type, permute_length):
        if permute_type:
            res_weight = self.res_weights[res_weight_id]
            if permute_type == True:
                if permute_length != None:
                    x_skip = torch.mean(x[:, :permute_length, :], dim=1, keepdim=True)
                else:
                    x_skip = torch.mean(x, dim=1, keepdim=True)
                all_zeros = torch.zeros_like(x)
                all_zeros[:, 0:1, :] += res_weight* x_skip
                x_skip = all_zeros
            elif permute_type == 'shuffle':
                if x.size(1) > permute_length:
                    randx = self.randx[:x.size(1)]
                    x_skip = res_weight* x[:, randx, :]
                else:
                    x_skip = res_weight* x
            elif permute_type == 'remove':
                if x.size(1) > permute_length:
                    x_skip = torch.concatenate([torch.repeat_interleave(self.voids, x.size(0), dim=0), x[:, permute_length:, :]], dim=1)
                else:
                    x_skip = torch.repeat_interleave(self.voids, x.size(0), dim=0)[:, :x.size(1), :]
                x_skip = res_weight* x_skip    
        else: # go normally
            x_skip = x
        return x_skip 


    def forward(self, 
                x, 
                attn_mask=None, 
                ablation_config=dict() # this will only be used during visualization for testing different operations
                ):



        if self.layerwise_pe:
            if self.pe_type == 'original':
                device = x.device
                b, t, d = x.size()
                assert t <= self.block_size, f"Block {self.id} cannot forward sequence of length {t}, block size is only {self.block_size}"
                pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

                pos_emb = self.layer_wpe(pos) # position embeddings of shape (1, t, n_embd)
                x = x + pos_emb # no dropout here
            elif self.pe_type == 'sin':
                x = self.layer_wpe(x)
        
        ln_1_fn = lambda x:x if ablation_config.get('no_ln_1', False) else self.ln_1(x)

        if self.no_att_residual:
            x = self.attn(ln_1_fn(x), attn_mask=attn_mask, ablation_config=ablation_config)

        else:
            x_skip = self.permute_func(x, res_weight_id=0)
            x = x_skip + self.attn(ln_1_fn(x), attn_mask=attn_mask, ablation_config=ablation_config)
            # if self.permute:
      
            #     if self.permute == True:
            #         if self.permute_length != None:
            #             x_skip = torch.mean(x[:, :self.permute_length, :], dim=1, keepdim=True)
            #         else:
            #             x_skip = torch.mean(x, dim=1, keepdim=True)
            #         all_zeros = torch.zeros_like(x)
            #         all_zeros[:, 0:1, :] += self.res1_weight * x_skip
            #         x_skip = all_zeros
            #     elif self.permute == 'shuffle':
            #         if x.size(1) > self.permute_length:
            #             randx = self.randx[:x.size(1)]
            #             x_skip = self.res1_weight * x[:, randx, :]
            #         else:
            #             x_skip = self.res1_weight * x
            #     elif self.permute == 'remove':
            #         if x.size(1) > self.permute_length:
            #             x_skip = torch.concatenate([torch.repeat_interleave(self.voids, x.size(0), dim=0), x[:, self.permute_length:, :]], dim=1)
            #         else:
            #             x_skip = torch.repeat_interleave(self.voids, x.size(0), dim=0)[:, :x.size(1), :]
            #         x_skip = self.res1_weight * x_skip    

            #     x = x_skip + self.attn(ln_1_fn(x), attn_mask=attn_mask, ablation_config=ablation_config)
            # else: # go normally
            #     x = x + self.attn(ln_1_fn(x), attn_mask=attn_mask, ablation_config=ablation_config)

            
            # randx = torch.randperm(x.size(1)).to(x.device)
            # x = x[:, self.randx, :]


        ln_2_fn = lambda x:x if ablation_config.get('no_ln_2', False) else self.ln_2(x)
        mlp_fn = lambda x:x if ablation_config.get('no_mlp', False) else self.mlp(x)

        if self.no_mlp_residual:
            x = mlp_fn(ln_2_fn(x))
        else:
            x_skip = self.permute_func(x, res_weight_id=1)
            x = x_skip + mlp_fn(ln_2_fn(x))
            # if self.permute:
            #     # randx = self.randx[x.size(1)].to(x.device)
            #     # x_skip = torch.mean(x, dim=1, keepdim=True)
            #     # x_skip = self.res2_weight * torch.repeat_interleave(x_skip, x.size(1), dim=1)

            #     ## all goes to the first
            #     if self.permute == True:
            #         x_skip = torch.mean(x, dim=1, keepdim=True)
            #         all_zeros = torch.zeros_like(x)
            #         all_zeros[:, 0:1, :] += self.res2_weight * x_skip
            #         x_skip = all_zeros
            #     elif self.permute == 'shuffle':
            #         if x.size(1) > self.permute_length:
                        
            #             randx = self.randx[:x.size(1)]
            #             x_skip = self.res2_weight * x[:, randx, :]
            #         else:
            #             x_skip = self.res2_weight * x
            #     elif self.permute == 'remove':
            #         if x.size(1) > self.permute_length:
            #             x_skip = torch.concatenate([torch.repeat_interleave(self.voids, x.size(0), dim=0), x[:, self.permute_length:, :]], dim=1)
            #         else:
            #             x_skip = torch.repeat_interleave(self.voids, x.size(0), dim=0)[:, :x.size(1), :]
            #         x_skip = self.res2_weight * x_skip    

            #     x = x_skip + mlp_fn(ln_2_fn(x))
            # else:
            #     x = x + mlp_fn(ln_2_fn(x)) # original
        
        return x

from typing import Any


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    use_flash: bool = True # use Flash Attention CUDA kernels for fast attention
    use_residual: Any = True
    no_att_residual: Any = False
    no_mlp_residual: Any = False 
    use_pe: str = 'original'
    layerwise_pe: bool = False
    layer_pe: str = 'original' # [original, sin]
    permute: bool = False
    permute_length: int = None
    not_causal: bool = False
    # add a deterministic option...
    

def handle_list(config, key):
    if hasattr(config, key):
        attr = config.__getattribute__(key)
        attr_per_layer = np.zeros(config.n_layer)
          
        if type(attr) is list:
            if str in [type(a) for a in attr]:
                assert len(attr) == config.n_layer, f"Length of {key} should be equal to n_layer"
                attr_per_layer = attr
            else:
                for n in attr:
                    attr_per_layer[n] += 1 # this allows double skip for residual connections
        elif type(attr) is int:
            attr_per_layer[:attr] = 1
        else:
            assert type(attr) is bool, f" {attr} {type(attr)} need to have id to use integer or "
            attr_per_layer = attr_per_layer + int(attr)
    return list(attr_per_layer)


import numpy as np
class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None

        config.use_residual = handle_list(config, 'use_residual')
        config.no_att_residual = handle_list(config, 'no_att_residual')
        config.no_mlp_residual = handle_list(config, 'no_mlp_residual')
        config.layerwise_pe = handle_list(config, 'layerwise_pe')
        config.permute = handle_list(config, 'permute')
        config.not_causal = handle_list(config, 'not_causal')


        if not hasattr(config, 'use_pe'):
            config.use_pe = 'nope'

        if config.use_pe == 'nope': 
            config.layerwise_pe = [0]*config.n_layer     


        transformer = dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            # wpe = nn.Embedding(config.block_size, config.n_embd), ## Nope
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config, id=id) for id in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        )

        if config.use_pe == 'original':
            transformer['wpe'] = nn.Embedding(config.block_size, config.n_embd)
        elif config.use_pe == 'nope':
            pass
        elif config.use_pe == 'sin':
            transformer['wpe'] = SinusoidalPositionalEncoding(config.n_embd, config.block_size)
                    
        print(f'PE in use: {config.use_pe}')
        self.transformer = nn.ModuleDict(transformer)


        self.config = config
        
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate

        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying
        # self.none_use_weights = self.lm_head.weight
        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))
        print('test_run')
        with torch.no_grad():
            idx = torch.ones(1, config.block_size).long()
            self(idx, debug=True) # dummy forward call to create parameter buffers

        print(self)

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        ## Nope
        if non_embedding:
            # n_params -= self.transformer.wte.weight.numel()
            if hasattr(self.transformer, 'wpe'):
                if hasattr(self.transformer.wpe, 'weight'):
                    n_params -= self.transformer.wpe.weight.numel() 
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @staticmethod
    def create_equal_distancing_vecotrs(n, dim, small_component=0.1): # the larger the small component, the closer the vectors
        # Initialize an array to store the vectors
        vectors = np.zeros((n, dim))
        one_locations = np.random.permutation(dim)[:n]
        # Generate orthogonal vectors (for simplicity, using an identity matrix for the first n vectors)
        for row, col in enumerate(one_locations):
            vectors[row, col] = 1
        
        

        # Normalize the vectors to have a norm of 1 (this step is optional and depends on the desired property)
        vectors = vectors / np.linalg.norm(vectors, axis=1)[:, np.newaxis]

        # Ensure equal pairwise dot product by adjusting the vectors
        # Adding a small component that is common to all vectors
        common_component = np.ones(dim) * small_component  # Small component added to ensure a non-zero dot product
        vectors += common_component


        # for i in range(len(vectors)):
        #     vectors[:-i] += common_component

        # Normalize again to ensure they're all of equal length (optional depending on requirements)
        vectors = vectors / np.linalg.norm(vectors, axis=1)[:, np.newaxis]
        # shuffle the vectors
        vectors = np.random.permutation(vectors) # yes pairwise equal distancing
        # vectors = np.random.rand(n, dim) # yes, because rand actually allows close to pairwise equal distancing
        # vectors = vectors[0:1].repeat(n, axis=0) + np.arange(n)[..., None] * 0.1
        # Verify the pairwise dot products
        pairwise_dot_products = np.dot(vectors, vectors.T)

        return vectors, pairwise_dot_products



    def forward(self, 
                idx, 
                targets=None, # put a target here during training
                debug=False,  
                causal_training=True, # the original trainijng method is causal, meaning the whole shifted input will be used as target 
                attn_mask=None, # for non-causal auto-regressive training that inputs non-equal length senquence

                equal_distancing_exp=False, # for visualization on equal distancing vectors
                direct_input_modification=False, # let the input skip the embedding and directly go to the first attention layer, for visualization
                ablation_config=dict()): # for attention ablation study
        '''
        Forward function for the GPT model
        
        Parameters:
        - idx: input token indices, shape [b, t]
        - targets: output token indices, shape [b, t]
        - debug: whether to print debug information
        - causal_training: whether to train in a causal or non-causal manner, both are auto-regressive. 
        Causal training the original way to train causal transformer, but to compare with non-causal transformer, we set this to be False 
        - attn_mask: attention mask for non-causal auto-regressive training
        - equal_distancing_exp: whether to use equal distancing vectors
        - ablation_config: configuration for attention ablation study

        Returns:
        - logits: output logits, shape [b, t, vocab_size]
        - loss: loss value, only calculated when targets are provided
        '''


        # forward the GPT model itselfs

        if not direct_input_modification:
            if equal_distancing_exp: # override the input idx with equal distancing vectors
                L = idx.size(1)
                vecs, dist_mat = GPT.create_equal_distancing_vecotrs(L, self.config.n_embd)
                x = torch.tensor(vecs[None, ], dtype=torch.float32).to(idx.device)
                
            else: # go normally
                ## Nope
                tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
                if self.config.use_pe == 'original':
                    device = idx.device
                    b, t = idx.size()
                    assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
                    pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)
                    pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
                    x = tok_emb + pos_emb
                elif self.config.use_pe == 'sin':
                    x = self.transformer.wpe(tok_emb)
                else:
                    x = tok_emb

            x = self.transformer.drop(x)
        else:
            x = idx # the idx now should be a sequence of vectors

        temp_counter = {}
        for bidx, block in enumerate(self.transformer.h):
            if debug:
                print(bidx, end=' ')
            x = block(x, attn_mask=attn_mask, ablation_config=ablation_config)
            for tcidx in temp_counter:
                if temp_counter[tcidx]['skip_count'] >= 0: # bug here (used to be >), now fixed
                    temp_counter[tcidx]['skip_count'] -= 1
                if temp_counter[tcidx]['skip_count'] == 0:
                    x = x + temp_counter[tcidx]['x']
                    if debug:
                        print(f'[{tcidx}]', end=' ')
            if debug:
                print()
            # when use residual are false, 0-1=-1 < 0, which will not trigger this. 
            # Only when it is something larger than 1, will this number determine which layer the current layer will skip to
            skip_count = self.config.use_residual[bidx] - 1 
            if skip_count > 0:
                temp_counter[bidx] = {}
                temp_counter[bidx]['skip_count'] = skip_count
                temp_counter[bidx]['x'] = x 
                
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            if causal_training:
                logits = self.lm_head(x)
            else:
                logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            # print(logits.view(-1, logits.size(-1)).shape, targets.view(-1).shape)
            # print(logits.view(-1, logits.size(-1)).argmax(-1)[:10], targets.view(-1)[:10])
        else:
            # print('check what is used to predict', x.shape) 
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        ## Nope
        if hasattr(self.transformer.wpe, 'weight'):
            self.transformer.wpe = nn.Embedding(block_size, self.config.n_embd)

        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        # assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        ## Nope missing one key
        # assert len(sd_keys_hf)-1 == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"

        for k in sd_keys_hf:
            if 'wpe' in k: continue # Nope ignore position embedding parameters 
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        print('calling from pretrained')
        print(model)

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)
                # elif isinstance(m, torch.nn.Parameter):
                #     no_decay.add(fpn)
                elif pn.endswith('_weight') or pn.endswith('freqs') or 'RelativePositionBias' in pn: # for res1, res2 weight params; 'freqs' for rotary embeddings
                    no_decay.add(fpn)

                #     print('mn:', mn)
                #     print('pn:', pn)
                #     print(pn.endswith('weight'))
                #     print(isinstance(m, blacklist_weight_modules))

        # subtle: 'transformer.wte.weight' and 'lm_head.weight' are tied, so they
        # will appear in the no_decay and decay sets respectively after the above.
        # In addition, because named_parameters() doesn't return duplicates, it
        # will only return the first occurence, key'd by 'transformer.wte.weight', below.
        # so let's manually remove 'lm_head.weight' from decay set. This will include
        # this tensor into optimization via transformer.wte.weight only, and not decayed.
        decay.remove('lm_head.weight')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        ## Nope doesn't have to worry about this
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        # new PyTorch nightly has a new 'fused' option for AdamW that is much faster
        use_fused = (device_type == 'cuda') and ('fused' in inspect.signature(torch.optim.AdamW).parameters)
        print(f"using fused AdamW: {use_fused}")
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, causal=True, attn_mask=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """

        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond, attn_mask=attn_mask)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)
            
            # padd true around attn mask
            if attn_mask is not None:
                new_mask = torch.ones(attn_mask.shape[0], attn_mask.shape[1]+1, attn_mask.shape[2]+1).bool()
                new_mask[:, :-1, :-1] = attn_mask
                attn_mask = new_mask.to(attn_mask.device)

  
        return idx
    
    def autoregressive_training(self, idx, targets, answer_mask, max_new_tokens, attn_mask=None):
        """
        Performs autoregressive training on the model.
        """
        loss = 0.0
        logits_list = []
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond, attn_mask=attn_mask)
            logits_list.append(logits)
            logits = logits[:, -1, :]
            # apply softmax to convert logits to (normalized) probabilities

            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)
            
            # padd true around attn mask
            if attn_mask is not None:
                new_mask = torch.ones(attn_mask.shape[0], attn_mask.shape[1]+1, attn_mask.shape[2]+1).bool()
                new_mask[:, :-1, :-1] = attn_mask
                attn_mask = new_mask.to(attn_mask.device)

        # Calculate loss with the logits and the shifted targets
        out_logits = torch.cat(logits_list, dim=1)
        answer_mask = answer_mask.flatten().bool()
        # loss = F.cross_entropy(out_logits.view(-1, out_logits.size(-1))[:, answer_mask], targets.view(-1)[answer_mask], ignore_index=-1)
 
        loss = F.cross_entropy(out_logits.view(-1, out_logits.size(-1))[answer_mask, ...], targets.view(-1)[answer_mask], ignore_index=-1)
        
        return out_logits, loss