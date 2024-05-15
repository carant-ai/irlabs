from torch import nn, Tensor
import torch
from typing import Optional, Dict, Literal


class MarginMSE(nn.Module):
    def __init__(
        self,
        # TODO: add scoring functin params
        # scoring_function: Literal["cos_sim", "dot_sim"] = "cos_sim",
        features_mapping: Optional[Dict[str, str]] = None,
        labels_mapping: Optional[Dict[str, str]] = None,
    ):
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
        super().__init__()

    def forward(self, batch:Dict[str, Tensor]):
        pos_score = (
            batch[self.features_mapping["anchor"]]
            * batch[self.features_mapping["positive"]]
        ).sum(dim=-1)

        neg_score = (
            batch[self.features_mapping["anchor"]]
            * batch[self.features_mapping["negative"]]
        ).sum(dim=-1)

        pred_diff = pos_score - neg_score
        labels_diff = batch[self.label_mapping["positive_score"]] - batch[self.label_mapping["negative_score"]]

        return self.mse_loss(pred_diff, labels_diff)
