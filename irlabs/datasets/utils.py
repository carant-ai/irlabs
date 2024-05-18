from datasets import Dataset, DatasetDict
from irlabs.models.config import IRConfig
from transformers import PretrainedConfig, AutoTokenizer
from typing import Any, Dict, Optional, List


def preprocess_tokenize_single_loader(
    datasets: Dataset | DatasetDict,
    config: PretrainedConfig,
    features: List[str],
    num_proc: int,
    save_file: str,
    mapping_kwargs: Optional[Dict[str, Any]] = None,
):
    tokenizer = AutoTokenizer.from_pretrained(config.name_or_path)
    if mapping_kwargs is None:
        mapping_kwargs = {}

    def tokenize(batch):
        new_batch = {}
        for feature in features:
            new_batch[feature] = tokenizer(
                batch[feature],
                max_length = config.max_d_length,
                padding = "max_length",
                truncation = True, 
                return_tensors = "pt"
            )
        return _flatten_features(new_batch)

    return datasets.map(tokenize, **mapping_kwargs, remove_columns= features, num_proc = num_proc, cache_file_name= save_file)



def _flatten_features(features):
    flattened_features = {}
    for feat in features:
        for k, v in features[feat].items():
            flattened_features[f"{feat}_{k}"] = v

    return flattened_features
