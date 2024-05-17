from dataclasses import dataclass
from lightning import LightningModule


from typing import Dict, Any

from transformers import PreTrainedModel, PretrainedConfig
from irlabs.optimizers import LinearDecayWithWarmupScheduler, optimizer_factory

from torch import nn


class IRModule(LightningModule):

    def __init__(
        self,
        model: PreTrainedModel,
        loss_fn: nn.Module,
        optimizer_name: str,
        weight_decay: float,
        warmup_step: int,
        optimizer_hparams: Dict[str, Any]
    ):
        super().__init__()
        self.model =model
        self.loss_fn = loss_fn
        self.optimizer_name = optimizer_name
        self.weight_decay = weight_decay
        self.warmup_step = warmup_step
        self.optimizer_hparams = optimizer_hparams

    def step(self, batch, batch_idx, mode):
        features, labels = batch
        reps = {}
        for key, item in features.items():
            reps[key] = self.model(**item)

        print(f"DEBUGPRINT[14]: trainer.py:33: batch_feat_labels={reps}")
        loss = self.loss_fn(reps, labels)
        print(f"DEBUGPRINT[16]: trainer.py:39: loss={loss}")
        self.log(f"{mode}_{self.loss_fn._get_name()}", loss, on_step=True, on_epoch=True)
        return loss

    def training_step(self, *args: Any, **kwargs: Any):
        batch, batch_idx = args
        self.step(batch, batch_idx, mode="train")

    def validation_step(self, *args: Any, **kwargs: Any):
        batch, batch_idx = args
        self.step(batch, batch_idx, mode="val")

    def configure_optimizers(self):
        optimizer = optimizer_factory(
            self.model, self.optimizer_name, self.optimizer_hparams
        )
        scheduler = LinearDecayWithWarmupScheduler(
            optimizer, self.warmup_step, self.trainer.estimated_stepping_batches
        )
        return [optimizer], [scheduler]
