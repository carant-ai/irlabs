from transformers import PreTrainedModel, PreTrainedTokenizerBase, PretrainedConfig
from irlabs.models.config import IRConfig
import torch
from torch import nn
import logging
from typing import List

logger = logging.getLogger(__name__)

class CLSPooler(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inp, attention_mask):
        return inp.last_hidden_state[:, 0, :]



def build_pooling(c: PretrainedConfig):
    if c.ir_embed_pooling_strategy == "cls":
        return CLSPooler()
    else:
        return CLSPooler()
