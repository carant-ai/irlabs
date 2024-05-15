from transformers import AutoModel, PreTrainedModel, PretrainedConfig, BertModel
from lightning import LightningModule
from typing import Literal, Dict
from .utils import build_pooling
from .config import IRLabsConfig


class LMForEmbed(PreTrainedModel):
    config_class = IRLabsConfig

    def __init__(self, config: IRLabsConfig):
        super().__init__(config)
        self.lm = AutoModel(config)
        self.pooler = build_pooling(config)
        self.post_init()

    def forward(self, lm_input: Dict):
        model_output = self.lm(**lm_input)  # type: ignore
        return self.pooler(model_output)  # type: ignore
