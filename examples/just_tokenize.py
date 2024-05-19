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
    config = IRConfig(ir_q_prefix="babidong:")
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

    data_module = SingleLoaderModule(
        dataset,
        "/mnt/disks/persist/.cache/indo_40M",
        model.config,
        ["positive", "anchor", "negative"],
        ["positive_score", "negative_score"],
        val_ratio=0.01,
        num_workers = 8,
        num_procs = 16,
    )

    data_module.prepare_data()

if __name__ == "__main__":
    main()
