from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import WandbLogger
from torch._prims_common import infer_size_shapes
from irlabs.models import IRConfig, BertForEmbedding
from transformers import AutoTokenizer
from irlabs.trainer import IRModule
from irlabs.losses.MarginMSE import MarginMSE
from irlabs.datasets.SingleLoader import SingleLoaderModule
from torch import nn
from datasets import load_dataset, Dataset
from lightning import Trainer
import os
import wandb
from lightning.pytorch import seed_everything
import lightning.pytorch as pl



def main():

    seed_everything(42, workers=True)
    wandb.login()
    model = BertForEmbedding.from_pretrained("indobenchmark/indobert-base-p1")
    dataset = load_dataset(
        "csv",
        data_files="/mnt/disks/persist/yourfile.tsv",
        sep="\t",
        split="train[:20%]",
    )

    if not isinstance(model, BertForEmbedding):
        raise
    if not isinstance(dataset, Dataset):
        raise

    data_module = SingleLoaderModule(
        dataset,
        "/mnt/disks/persist/loaded/new",
        model.config,
        ["positive", "anchor", "negative"],
        ["positive_score", "negative_score"],
        val_ratio=0.01,
        batch_size=32,
        num_workers=16,
        drop_last=True,
        shuffle=True,
    )

    optimizers_hparams = {
        "lr": 2e-6,
    }

    ir_module = IRModule(
        model=model,
        tokenizer = data_module.tokenizer,
        loss_fn=MarginMSE(),
        optimizer_name="Adam",
        weight_decay=5e-4,
        warmup_step=0.1,
        optimizer_hparams=optimizers_hparams,
        features = ["positive", "anchor", "negative"],
        labels = ["positive_score", "negative_score"],
    )

    logger = WandbLogger(
        "indo-embed-mmarco", save_dir="/mnt/disks/persist/train_artifact/", log_model="all", project = "indo-ir-gaps"
    )

    model_checkpoint = ModelCheckpoint(monitor="val_loss", mode="min",)
    lr_monitor = LearningRateMonitor("step")
    trainer = Trainer(
        callbacks=[ model_checkpoint, lr_monitor],
        logger=logger,
        max_epochs=1,
    )
    trainer.fit(ir_module, data_module)


if __name__ == "__main__":
    main()
