from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
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


def main():

    wandb.login()
    model = BertForEmbedding.from_pretrained("indobenchmark/indobert-base-p1")
    dataset = load_dataset(
        "csv",
        data_files="/mnt/disks/persist/yourfile.tsv",
        sep="\t",
        split="train[:10%]",
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
        False,
        val_ratio=0.01,
        num_workers = 32,
    )

    optimizers_hparams = {
        "lr": 2e-5,
    }

    ir_module = IRModule(
        model=model,
        loss_fn=MarginMSE(),
        optimizer_name="Adam",
        weight_decay=5e-4,
        warmup_step=0.1,
        optimizer_hparams=optimizers_hparams,
    )

    logger = WandbLogger("indo-embed-40M", save_dir = "/mnt/disks/persist/train_artifact/", log_model = "all")
    early_stopping = EarlyStopping(monitor= "val_loss", log_rank_zero_only= True, min_delta = 1e-4)
    model_checkpoint = ModelCheckpoint(monitor = "val_loss", mode = "min")
    lr_monitor = LearningRateMonitor("step")
    trainer = Trainer(callbacks=[early_stopping, model_checkpoint, lr_monitor], logger = logger, max_epochs = 4)
    trainer.fit(ir_module, data_module)


if __name__ == "__main__":
    main()
