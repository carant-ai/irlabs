from irlabs.models.embed import LMForEmbedConfig
import torch
from torch import nn
# i'll improve it later with enum 

class CLSPooler(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inp):
        return inp.last_hidden_state[:, 0, :]

def build_pooling(c: LMForEmbedConfig):
    if c.pooling_strategy == "max":
        raise NotImplementedError 

    elif c.pooling_strategy == "cls":
        return CLSPooler()

    elif c.pooling_strategy == "mean":
        raise NotImplementedError
