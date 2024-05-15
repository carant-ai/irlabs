from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

import torch


@dataclass
class SingleLoaderCollator:
    valid_features: List[str]
    valid_labels: List[str] | None
    model_input_names: List[str]

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        columns = list(batch[0].keys())
        new_batch = {}
        for feature in self.valid_features:
            new_batch[f"{feature}"] = {
                    k: torch.cat(
                    [row[f"{feature}_{k}"].view(1, -1) for row in batch], dim=0
                ) for k in (self.model_input_names)
            }

        if self.valid_labels is None:
            return new_batch

        for label_column in self.valid_labels:
            new_batch[label_column] = torch.cat(
                [row[label_column].view(1, -1) for row in batch], dim=0
            ).view(-1)

        return new_batch
