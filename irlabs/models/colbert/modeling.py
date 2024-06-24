from typing import Optional
import torch
import transformers
from irlabs.models.config import IRConfig, resolve_config
from torch import nn
from torch.nn import functional as F
import logging

logger = logging.getLogger(__name__)
import transformers.modeling_outputs


class BertForColbert(transformers.BertPreTrainedModel):
    config_class = transformers.BertConfig

    def __init__(
        self,
        config: transformers.BertConfig,
        ir_config: Optional[IRConfig] = None,
    ):
        super().__init__(config)
        combined_config = resolve_config(ir_config, config, "colbert")

        self.bert = transformers.BertModel(config, add_pooling_layer=False)
        self.config = IRConfig.from_dict(combined_config)
        self.linear = nn.Linear(
            self.config.hidden_size, self.config.colbert_embedding_size, bias=False
        )
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:

        output = self.bert(input_ids, attention_mask, token_type_ids, **kwargs)
        assert isinstance(
            output,
            transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions,
        ), "Expected 'output' to be of type 'transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions', but got '{}' instead.".format(
            type(output).__name__
        )

        return (
            F.normalize(
                self.linear(output.last_hidden_state * attention_mask.unsqueeze(-1)),
                dim=2,
                p=2,
            )
            if self.config.normalize
            else self.linear(output.last_hidden_state * attention_mask.unsqueeze(-1))
        )
