from typing import Optional
import torch
import transformers
from irlabs.models.config import IRConfig, resolve_config
from .utils import build_pooling
import logging
import transformers.modeling_outputs

logger = logging.getLogger(__name__)


class BertForEmbedding(transformers.BertPreTrainedModel):
    config_class = transformers.BertConfig

    def __init__(
        self,
        config: transformers.BertConfig,
        ir_config: Optional[IRConfig] = None,
    ):
        super().__init__(config)
        combined_config = resolve_config(ir_config, config, "embed")
        self.bert = transformers.BertModel(config, add_pooling_layer=False)
        self.config = IRConfig.from_dict(combined_config)
        self.pooler = build_pooling(self.config)
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:

        output = self.bert(input_ids, attention_mask, token_type_ids, position_ids)

        assert isinstance(
            output,
            transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions,
        ), "Expected 'output' to be of type 'transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions', but got '{}' instead.".format(
            type(output).__name__
        )
        return self.pooler(output, attention_mask=attention_mask)
