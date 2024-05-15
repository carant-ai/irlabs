from datasets import Dataset, IterableDataset
from irlabs.models.config import IRLabsConfig
from transformers import PretrainedConfig, AutoTokenizer
from typing import Any, Dict, Optional, List


def preprocess_tokenize_single_loader(
    datasets: Dataset,
    config: IRLabsConfig,
    features: List[str],
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

    datasets = datasets.map(tokenize, **mapping_kwargs, remove_columns= features)
    datasets.set_format("torch", features)
    return datasets



def _flatten_features(features):
    flattened_features = {}
    for key in features:
        for k, v in features[key].items():
            flattened_features[f"{key}_{k}"] = v

    return flattened_features
