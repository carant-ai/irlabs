from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Tuple, Optional
from transformers import PreTrainedTokenizerBase, PretrainedConfig
import torch

def _flatten_features(features):
    flattened_features = {}
    for feat in features:
        for k, v in features[feat].items():
            flattened_features[f"{feat}_{k}"] = v

    return flattened_features

@dataclass
class SingleLoaderCollator:
    valid_features: List[str]
    valid_labels: List[str] | None
    model_input_names: List[str]


    def __call__(
        self, batch: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]]]:
        new_batch = {}
        new_batch_labels = {}
        for feature in self.valid_features:
            new_batch[f"{feature}"] = {
                k: torch.cat(
                    [row[f"{feature}_{k}"].view(1, -1) for row in batch], dim=0
                )
                for k in (self.model_input_names)
            }

        if self.valid_labels is None:
            return new_batch, None

        for label_column in self.valid_labels:
            new_batch_labels[label_column] = torch.cat(
                [row[label_column].view(1, -1) for row in batch], dim=0
            ).view(-1)

        return new_batch, new_batch_labels


@dataclass
class SingleLoaderCollatorWithTokenize:
    valid_features: List[str]
    valid_labels: List[str] | None
    tokenizer: PreTrainedTokenizerBase
    config: PretrainedConfig

    def __call__(
        self, batch: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]]]:
        new_batch = {}
        new_batch_labels = {}

        for feat in self.valid_features:
            new_batch[feat] = self.tokenizer(
                [row[feat] for row in batch],
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

        if new_batch_labels["positive_score"].shape != (16,) or new_batch_labels["negative_score"].shape != (16,):
            print(f"DEBUGPRINT[7]: collator.py:74: self.valid_labels={self.valid_labels}")
            print(f"DEBUGPRINT[8]: collator.py:75: self.valid_features={self.valid_features}")
            print(f"DEBUGPRINT[1]: collator.py:68: new_batch_labels={new_batch_labels}")

        return new_batch, new_batch_labels
