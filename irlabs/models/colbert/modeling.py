from typing import Optional
import torch
from transformers import (
    BertConfig,
    BertModel,
    BertPreTrainedModel,
)
from irlabs.models.config import IRConfig
from irlabs.models.utils import combine_dict
from torch import nn
from torch.nn import functional as F
import logging

logger = logging.getLogger(__name__)
from transformers.modeling_outputs import BaseModelOutputWithCrossAttentions


class BertForColbert(BertPreTrainedModel):
    config_class = BertConfig

    def __init__(
        self,
        config: BertConfig,
        ir_config: Optional[IRConfig] = None,
    ):
        super().__init__(config)
        ir_config_dict = {} if not ir_config else ir_config.to_dict()

        if not ir_config and hasattr(config, "is_ir_config"):
            logger.info(
                "ir_config parameter is None and config is an instance of IRConfig."
            )
        elif not ir_config and not hasattr(config, "is_ir_config"):
            logger.info(
                "ir_config parameter is None and config is not an instance of IRConfig. Loading default IRConfig."
            )
            ir_config_dict = IRConfig("colbert").to_dict()
        elif ir_config and hasattr(config, "is_ir_config"):
            logger.info(
                "ir_config is not None and config is an instance of IRConfig. Replacing older IRConfig related attributes from ir_config."
            )
            ir_config_dict = ir_config.to_dict()

        self.bert = BertModel(config, add_pooling_layer=False)
        self.config = IRConfig.from_dict(combine_dict(config.to_dict(), ir_config_dict))
        self.linear = nn.Linear(
            self.config.hidden_size, self.config.colbert_embedding_size, bias=False
        )
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        position_ids: torch.Tensor,
        **kwargs,
    ) -> Optional[torch.Tensor]:

        output = self.bert(
            input_ids, attention_mask, token_type_ids, position_ids, **kwargs
        )

        assert isinstance(output, BaseModelOutputWithCrossAttentions)

        return (
            F.normalize(
                self.linear(output.last_hidden_state * attention_mask.unsqueeze(-1)),
                dim=2,
                p=2,
            )
            if self.config.normalize
            else self.linear(output.last_hidden_state * attention_mask.unsqueeze(-1))
        )
