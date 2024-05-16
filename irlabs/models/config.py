from transformers import PretrainedConfig
from typing import Literal

class IRConfig(PretrainedConfig):
    def __init__(
        self,
        max_q_length: int = 128,
        max_d_length: int = 128,
        q_prefix: str = "",
        d_prefix: str = "", 
        embed_pooling_strategy: Literal["mean", "max", "cls", None]  = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_q_length = max_q_length
        self.max_d_length = max_d_length
        self.q_prefix = q_prefix
        self.d_prefix = d_prefix
        self.pooling_strategy = embed_pooling_strategy 
