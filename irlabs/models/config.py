from transformers import PretrainedConfig
from typing import Literal

class IRConfig(PretrainedConfig):
    def __init__(
        self,
        ir_max_q_length: int = 128,
        ir_max_d_length: int = 128,
        ir_q_prefix: str = "",
        ir_d_prefix: str = "", 
        ir_embed_pooling_strategy: Literal["mean", "max", "cls", None]  = "cls",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.ir_max_q_length = ir_max_q_length
        self.ir_max_d_length = ir_max_d_length
        self.ir_q_prefix = ir_q_prefix
        self.ir_d_prefix = ir_d_prefix
        self.ir_embed_pooling_strategy = ir_embed_pooling_strategy 
