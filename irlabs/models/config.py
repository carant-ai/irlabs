from transformers import PretrainedConfig
import typing
import logger

def combine_dict(*args: typing.Dict):
    new_dict = {}
    for d in args:
        for key, value in d.items():
            if new_dict.get(key):
                continue

            new_dict[key] = value
    return new_dict



class IRConfig(PretrainedConfig):
    def __init__(
        self,
        ir_model_type: typing.Literal["embed", "colbert", "splade", None] = None,
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
        self.ir_model_type = ir_model_type
        self.is_ir_config = True
        self.ir_max_q_length = ir_max_q_length
        self.ir_max_d_length = ir_max_d_length
        self.ir_q_prefix = ir_q_prefix
        self.ir_d_prefix = ir_d_prefix
        self.normalize = normalize

        if self.ir_model_type == "embed":
            self.embed_pooling_strategy = embed_pooling_strategy
        elif self.ir_model_type == "colbert":
            self.colbert_embedding_size = colbert_embedding_size

def resolve_config(ir_config: typing.Optional[IRConfig], config: PretrainedConfig, model_type: str):
    ir_config_dict = {} if not ir_config else ir_config.to_dict()

    if not ir_config and hasattr(config, "is_ir_config"):
        logger.info(
            "ir_config parameter is None and config is an instance of IRConfig."
        )
    elif not ir_config and not hasattr(config, "is_ir_config"):
        logger.info(
            "ir_config parameter is None and config is not an instance of IRConfig. Loading default IRConfig."
        )
        ir_config_dict = IRConfig(model_type).to_dict()
    elif ir_config and hasattr(config, "is_ir_config"):
        logger.info(
            "ir_config is not None and config is an instance of IRConfig. Replacing older IRConfig related attributes from ir_config."
        )
        ir_config_dict = ir_config.to_dict()
    return combine_dict(config.to_dict(), ir_config_dict)
pass
