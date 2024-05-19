from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Tuple, Optional
from transformers import PreTrainedTokenizerBase, PretrainedConfig

import torch


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
            tokenized = self.tokenizer(
                [row[feat] for row in batch],
                max_length = self.config.ir_max_d_length,
                padding = "max_length",
                truncation = True, 
                return_tensors = "pt"
            )
            new_batch[feat] = tokenized

        if self.valid_labels is None:
            return new_batch, None

        for label_column in self.valid_labels:
            new_batch_labels[label_column] = torch.cat(
                [row[label_column].view(1, -1) for row in batch], dim=0
            ).view(-1)

        return new_batch, new_batch_labels
