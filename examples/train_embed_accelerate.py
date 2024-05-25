from accelerate import Accelerator
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
from torch import nn, optim
from datasets import load_dataset, Dataset
from lightning import Trainer
import os
import wandb
from tqdm import tqdm


def main():
    model = BertForEmbedding.from_pretrained("indobenchmark/indobert-base-p1")
    accelerator = Accelerator()
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
        val_ratio=0.01,
        batch_size=16,
        num_workers=8,
        drop_last=True,
        shuffle=False,
    )

    data_module.setup(stage="fit")
    optimizer = optim.Adam(model.parameters(), lr=2e-5)
    loss_fn = MarginMSE()
    model, optimizer, training_dataloader = accelerator.prepare(
        model, optimizer, data_module.train_dataloader()
    )

    for epoch in tqdm(range(1)):
        for features, labels in tqdm(training_dataloader):
            optimizer.zero_grad()
            reps = {}
            for key in ["positive", "negative", "anchor"]:
                reps[key] = model.forward(
                    input_ids = features[f"{key}_input_ids"],
                    attention_mask = features[f"{key}_attention_mask"],
                    token_type_ids = features[f"{key}_token_type_ids"],
                )

            loss = loss_fn(reps, labels, features)
            accelerator.backward(loss)
            optimizer.step()

    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(
        "/mnt/disks/persist/model/embacc",
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
    )


if __name__ == "__main__":
    main()
