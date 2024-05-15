from dataclasses import dataclass
from irlabs.models.embed import PreTrainedModel
from lightning import LightningModule


from typing import Dict, Any
from irlabs.optimizers import LinearDecayWithWarmupScheduler, optimizer_factory

from torch import nn


@dataclass
class IRTrainerModule(LightningModule):
    model: PreTrainedModel
    loss_fn: nn.Module
    optimizer_name: str
    weight_decay: float
    warmup_step: int
    optimizer_hparams: Dict[str, Any]

    def step(self, batch, batch_idx, mode):
        batch_feat_labels = {}
        for key, item in batch.items():
            batch_feat_labels[key] = self.model(**batch)
        loss = self.loss_fn(batch_feat_labels)
        self.log(f"{mode}_{self.loss_fn.__name__}", loss, on_step=True, on_epoch=True)
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
