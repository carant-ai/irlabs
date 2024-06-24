import torch
from torch import nn
from typing import Callable, Optional, Dict, Literal
from .utils import resolve_scoring_function


class MarginMSE(nn.Module):
    def __init__(
        self,
        scoring_function: (
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
            | Literal["dot", "maxsim"]
        ),
        features_mapping: Optional[Dict[str, str]] = None,
        labels_mapping: Optional[Dict[str, str]] = None,
    ):
        super().__init__()
        if isinstance(scoring_function, str):
            self.scoring_function = resolve_scoring_function(scoring_function)
        else:
            self.scoring_function = scoring_function

        if features_mapping is None:
            self.features_mapping = {
                "anchor": "anchor",
                "positive": "positive",
                "negative": "negative",
            }
        if labels_mapping is None:
            self.label_mapping = {
                "positive_score": "positive_score",
                "negative_score": "negative_score",
            }
        self.mse_loss = nn.MSELoss()

    def forward(
        self,
        reps: Dict[str, torch.Tensor],
        labels: Dict[str, torch.Tensor],
    ):
        pos_score = self.scoring_function(
            reps[self.features_mapping["anchor"]],
            reps[self.features_mapping["positive"]],
        )

        neg_score = self.scoring_function(
            reps[self.features_mapping["anchor"]],
            reps[self.features_mapping["negative"]],
        )

        pred_diff = pos_score - neg_score
        labels_diff = (
            labels[self.label_mapping["positive_score"]]
            - labels[self.label_mapping["negative_score"]]
        )

        return self.mse_loss(pred_diff, labels_diff)
