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


def main():
    model = BertForEmbedding.from_pretrained("bert-base-uncased")
    dataset = load_dataset(
        "csv",
        data_files="/mnt/disks/persist/yourfile.tsv",
        sep="\t",
        split="train[:1%]",
        cache_dir="/mnt/disks/persist/huggingface/",
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
    )

    data_module.setup(stage="fit")
    data_loader = data_module.train_dataloader()

    optimizers_hparams = {
        "lr": 2e-5,
    }

    ir_module = IRModule(
        model=model,
        loss_fn=MarginMSE(),
        optimizer_name="Adam",
        weight_decay=5e-4,
        warmup_step=10000,
        optimizer_hparams=optimizers_hparams,
    )

    for batch_idx, batch in enumerate(data_loader):
        ir_module.training_step(batch, batch_idx)
        break
    return

    trainer = Trainer()
    trainer.fit(ir_module, train_dataloaders=data_loader)


if __name__ == "__main__":
    main()
