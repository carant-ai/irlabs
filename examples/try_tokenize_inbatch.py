from irlabs.models import IRConfig, BertForEmbedding
from transformers import AutoTokenizer
from irlabs.trainer import IRModule
from irlabs.losses.MarginMSE import MarginMSE
from irlabs.datasets.SingleLoader import HFSingleLoaderModule
from torch import nn
from datasets import load_dataset, Dataset
from lightning import Trainer
import os


def main():
    config = IRConfig()
    model = BertForEmbedding.from_pretrained("indobenchmark/indobert-base-p1", config)
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

    data_module = HFSingleLoaderModule(
        dataset,
        "/mnt/disks/persist/.cache/indo_40M",
        model.config,
        ["positive", "anchor", "negative"],
        ["positive_score", "negative_score"],
        tokenize_before = False,
        val_ratio=0.01,
        num_workers=8,
        batch_size = 3,
        num_procs=16,
    )

    data_module.prepare_data()
    data_module.setup(stage = "fit")
    for batch in data_module.train_dataloader():
        print(f"DEBUGPRINT[1]: try_tokenize_inbatch.py:42: batch={batch}")
        break


if __name__ == "__main__":
    main()
