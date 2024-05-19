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
from tqdm import tqdm
import torch


def main():

    model = BertForEmbedding.from_pretrained("indobenchmark/indobert-base-p1")
    dataset = load_dataset(
        "csv",
        data_files="/mnt/disks/persist/yourfile.tsv",
        sep="\t",
        split="train",
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
    )

    data_module.prepare_data()
    data_module.setup(stage = "fit")
    key = ["positive_score", "negative_score"]
    for features, labels in tqdm(data_module.train_dataloader()):
        diff = labels[key[0]] - labels[key[1]]
        assert diff.shape == (32,)

    for features, labels in tqdm(data_module.val_dataloader()):
        diff = labels[key[0]] - labels[key[1]]
        assert diff.shape == (32,)



if __name__ == "__main__":
    main()

