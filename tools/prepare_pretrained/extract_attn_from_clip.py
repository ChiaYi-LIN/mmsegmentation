import torch
import clip

for v in ['RN50', 'RN101']:
    model, preprocess = clip.load('RN50', 'cpu')
    in_dim = model.visual.attnpool.k_proj.in_features
    out_dim = model.visual.attnpool.c_proj.out_features
    torch.save(model.visual.attnpool.state_dict(), f'./pretrained/visualattnpool_{v}_{in_dim}_{out_dim}.pth')