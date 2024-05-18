from typing import Optional
import torch
from transformers import (
    BertConfig,
    BertModel,
    BertPreTrainedModel,
    AutoConfig,
    PretrainedConfig,
    is_ray_available,
)
from irlabs.models.config import IRConfig
from .utils import CLSPooler, build_pooling
import logging
logger = logging.getLogger(__name__)


class BertForEmbedding(BertPreTrainedModel):
    config_class = BertConfig

    def __init__(
        self,
        config: BertConfig,
        ir_config: Optional[IRConfig],
    ):
        super().__init__(config)
        ir_config_dict = {} if not ir_config else ir_config.to_dict()
        print(f"DEBUGPRINT[6]: modeling.py:25: config={config}")
        print(f"DEBUGPRINT[7]: modeling.py:27: ir_config_dict={ir_config_dict}")
        

        if not ir_config and hasattr(config, "is_ir_config"):
            logger.info("ir_config parameter is None and config is an instance of IRConfig.")
        elif not ir_config and not hasattr(config, "is_ir_config"):
            logger.info("ir_config parameter is None and config is not an instance of IRConfig. Loading default IRConfig.")
            ir_config_dict = IRConfig().to_dict()
        elif ir_config and hasattr(config, "is_ir_config"):
            logger.info("ir_config is not None and config is an instance of IRConfig. Replacing older IRConfig related attributes from ir_config.")
            ir_config_dict = ir_config.to_dict()

        self.bert = BertModel(config, add_pooling_layer=False)
        self.config = IRConfig.from_dict(config.to_dict(), **ir_config_dict)
        print(f"DEBUGPRINT[4]: modeling.py:38: self.config={self.config}")
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
        return self.pooler(output, attention_mask=attention_mask)

