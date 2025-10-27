from TestingGraphMemEfficient import ERGraphModel, make_biosemi64_info
import torch
from torchinfo import summary
n_ch = 64
blocks = 3
heads = 4
k = 8
device = "cpu"
_, _, pos = make_biosemi64_info()
model = ERGraphModel(n_ch=n_ch, pos=pos, d_stem=256, d_lift=127, d_in=128, d_model=128,
                         L=blocks, k=k, heads=heads, dropout=0.1, causal=True).to(device)

model.load_state_dict(torch.load("outputs/S1/best_model.pt", map_location=device))
summary(model)