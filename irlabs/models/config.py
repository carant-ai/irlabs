from transformers import PretrainedConfig
from typing import Literal

class IRConfig(PretrainedConfig):
    def __init__(
        self,
        model_type: Literal["embed", "colbert", "splade"],
        ir_max_q_length: int = 256,
        ir_max_d_length: int = 256,
        ir_q_prefix: str = "",
        ir_d_prefix: str = "", 
        embed_pooling_strategy: Literal["mean", "max", "cls", None]  = "cls",
        colbert_embedding_size: int = 128,
        normalize: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.is_ir_config = True
        self.ir_max_q_length = ir_max_q_length
        self.ir_max_d_length = ir_max_d_length
        self.ir_q_prefix = ir_q_prefix
        self.ir_d_prefix = ir_d_prefix
        self.normalize = normalize

        if model_type == "embed":
            self.embed_pooling_strategy = embed_pooling_strategy
        elif model_type == "colbert":
            self.colbert_embedding_size = colbert_embedding_size
