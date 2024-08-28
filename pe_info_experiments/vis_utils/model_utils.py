

import plotly.graph_objs as go
from sklearn.decomposition import PCA

import torch
import random


def load_checkpoint(ckpt_path, 
                    model_config, 
                    model_type, 
                    device='cuda', 
                    return_config=False, 
                    return_model=True,
                    init=False,
                    init_additional_config={},):
    # load ckpt into model
    checkpoint = torch.load(ckpt_path, map_location=device)

    model_args = checkpoint['model_args']
    original_gptconf = model_config(**model_args)
    gptconf = model_config(**model_args)
    if return_model:
        if not init:
            model = model_type(original_gptconf)
            
            state_dict = checkpoint['model']
            # fix the keys of the state dictionary :(
            # honestly no idea how checkpoints sometimes get this prefix, have to debug more
            unwanted_prefix = '_orig_mod.'
            for k, v in list(state_dict.items()):
                if k.startswith(unwanted_prefix):
                    state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            model.load_state_dict(state_dict)
        else:
            # override with keys
            for k in init_additional_config:
                original_gptconf.__dict__[k] = init_additional_config[k]
            model = model_type(original_gptconf)

    if return_config:
        if return_model:
            return model, gptconf
        else: 
            return gptconf
    else:
        return model


def generate_output(model, 
                    prompt, 
                    encode,
                    decode,
                    max_new_tokens=5, 
                    attn_mask=None, 
                    top_k=None,
                    device='cuda',
                    ):

    model.eval()
    model.to(device)
    if compile:
        model = torch.compile(model)  # requires PyTorch 2.0 (optional)

    start_ids = encode(prompt)
    x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

    with torch.no_grad():
        num_samples = 1
        for k in range(num_samples):

            attn_mask = None

            y = model.generate(x, max_new_tokens,
                            attn_mask=attn_mask, top_k=top_k)

    return decode(y[0].tolist())


def PCA_analysis(prompt, embs, out_text, config_dir):
    pca = PCA(n_components=2)
    new_x = pca.fit_transform(embs.cpu().numpy())
    data = []
    for i, (text, pt) in enumerate(zip(prompt, new_x)):
        trace = go.Scatter(
            x=[pt[0]],
            y=[pt[1]],
            mode='markers+text',
            marker=dict(size=10),  # Adjust the size of the points
            text=[str(i+1)],
            textposition='middle center',  # Center the text within the marker
            name=text,
            textfont=dict(
                family='Times New Rotman',  # Specify the font family
                size=18,  # Adjust the font size
                color='black',  # Adjust the font color
            ),
        )
        data.append(trace)

    layout = go.Layout(
        xaxis=dict(title='Principal Component 1'),
        yaxis=dict(title='Principal Component 2'),
        title=f'PCA visualization for {prompt}'
    )

    fig = go.Figure(data=data, layout=layout)
    fig.show()

    print(out_text)
    # print(new_x)
    print(pca.explained_variance_ratio_)
    import plotly.io as pio
    pio.write_html(fig, f'./{config_dir}/{prompt}.html')

