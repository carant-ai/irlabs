from dataclasses import dataclass
from lightning import LightningModule


from typing import Dict, Any, List

from transformers import PreTrainedModel, PreTrainedTokenizerBase, PretrainedConfig
from irlabs.optimizers import LinearDecayWithWarmupScheduler, optimizer_factory

from torch import nn


class IRModule(LightningModule):

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        loss_fn: nn.Module,
        optimizer_name: str,
        weight_decay: float,
        warmup_step: int | float,
        optimizer_hparams: Dict[str, Any],
        features: List[str],
        labels: List[str] | None,
    ):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer_name = optimizer_name
        self.weight_decay = weight_decay
        self.warmup_step = warmup_step
        self.optimizer_hparams = optimizer_hparams
        self.features = (features,)
        self.labels = labels
        self.tokenizer = tokenizer

    def step(self, batch, batch_idx, mode):
        features, labels = batch
        reps = {}
        for key in self.features:
            reps[key] = self.model(
                **{
                    f"{key}_{id}": features[f"{key}_{id}"]
                    for id in self.tokenizer.model_input_names
                }
            )

        loss = self.loss_fn(reps, labels)
        self.log(
            f"{mode}_{self.loss_fn._get_name()}", loss, on_step=True, on_epoch=True
        )
        return loss

    def training_step(self, *args: Any, **kwargs: Any):
        batch, batch_idx = args
        features, labels = batch
        loss = self.step(batch, batch_idx, mode="train")

        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, *args: Any, **kwargs: Any):
        batch, batch_idx = args
        loss = self.step(batch, batch_idx, mode="val")
        self.log("val_loss", loss, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        warmup_step = self.warmup_step
        if isinstance(warmup_step, float):
            warmup_step = self.trainer.estimated_stepping_batches * self.warmup_step

        optimizer = optimizer_factory(
            self.model, self.optimizer_name, self.optimizer_hparams
        )
        return [optimizer]
