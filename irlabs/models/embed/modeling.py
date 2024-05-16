from typing import Optional
import torch
from transformers import BertConfig, BertModel, BertPreTrainedModel
from irlabs.models.config import IRConfig
from .utils import build_pooling


class BertForEmbedding(BertPreTrainedModel):
    config_class = IRConfig

    def __init__(
        self,
        config: IRConfig,
    ):
        super().__init__(config)
        self.bert = BertModel(config, add_pooling_layer=False)
        self.pooler = build_pooling(config)
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:

        output = self.bert(input_ids, attention_mask, token_type_ids, position_ids)
        return self.pooler(output, attention_mask=attention_mask)
