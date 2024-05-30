from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Tuple, Optional
from transformers import PreTrainedTokenizerBase, PretrainedConfig
import torch

from irlabs.models.config import IRConfig

def _flatten_features(features):
    flattened_features = {}
    for feat in features:
        for k, v in features[feat].items():
            flattened_features[f"{feat}_{k}"] = v

    return flattened_features



@dataclass
class SingleLoaderCollatorWithTokenize:
    query_columns: List[str]
    document_columns: List[str]
    valid_labels: List[str] | None
    tokenizer: PreTrainedTokenizerBase
    config: IRConfig | PretrainedConfig

    def __call__(
        self, batch: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]]]:
        new_batch = {}
        new_batch_labels = {}

        for feat in self.query_columns:
            new_batch[feat] = self.tokenizer(
                [self.config.ir_q_prefix + row[feat] for row in batch],
                max_length = self.config.ir_max_q_length,
                padding = "max_length",
                truncation = True, 
                return_tensors = "pt"
            )

        for feat in self.document_columns:
            new_batch[feat] = self.tokenizer(
                [self.config.ir_d_prefix + row[feat] for row in batch],
                max_length = self.config.ir_max_d_length,
                padding = "max_length",
                truncation = True, 
                return_tensors = "pt"
            )

        new_batch = _flatten_features(new_batch)

        if self.valid_labels is None:
            return new_batch, None

        for label_column in self.valid_labels:
            new_batch_labels[label_column] = torch.tensor([row[label_column] for row in batch])


        return new_batch, new_batch_labels
