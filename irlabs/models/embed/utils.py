from irlabs.models.config import IRConfig
import torch
from torch import nn
# i'll improve it later with enum 

class CLSPooler(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inp, attention_mask):
        return inp.last_hidden_state[:, 0, :]

def build_pooling(c: IRConfig): 
    if c.embed_pooling_strategy == "cls":
        return CLSPooler()
    else: 
        return CLSPooler()




