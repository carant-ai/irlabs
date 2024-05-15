from posixpath import join
from irlabs.models.config import IRLabsConfig
from transformers import AutoModel, PreTrainedModel, PretrainedConfig, BertModel
from lightning import LightningModule
from irlabs.models.utils import build_pooling
from typing import Literal, Dict, Any


class LMForEmbed(PreTrainedModel, LightningModule):
    config_class = IRLabsConfig
    def __init__(
        self,
        config,
        training_config
    ):
        self.lm = AutoModel(config)
        self.pooler = build_pooling(config)
        self.post_init()

    def forward(self, lm_input: Dict):
        model_output = self.lm(**lm_input) #type: ignore
        return self.pooler(model_output) #type: ignore

    def training_step(self, *args: Any, **kwargs: Any): 
        return super().training_step(*args, **kwargs)


if __name__ =="__main__":
    model = LMForEmbed.from_pretrained("bert-base-uncased")

