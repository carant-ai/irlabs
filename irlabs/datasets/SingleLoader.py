from lightning import LightningDataModule
from torch.utils.data import DataLoader
import os


from typing import List, Any, Optional
from datasets import (
    DatasetDict,
    Dataset,
    concatenate_datasets,
)
import logging
from dataclasses import dataclass

from transformers import AutoTokenizer, PretrainedConfig
from .utils import preprocess_tokenize_single_loader
from .collator import SingleLoaderCollator, SingleLoaderCollatorWithTokenize

logger = logging.getLogger(__name__)


class SingleLoaderModule(LightningDataModule):
    def __init__(
        self,
        datasets: Dataset | DatasetDict,
        local_save_file: str,
        config: PretrainedConfig,
        features: List[str],
        labels: List[str] | None,
        tokenize_before: bool = False,
        val_ratio: float = 0.01,
        data_collator: Optional[Any] = None,
        batch_size: int = 32,
        seed: int = 42,
        drop_last: bool = False,
        shuffle: bool = True,
        num_workers: int = 8,
        pin_memory: bool = True,
        persistent_workers: bool = False,
        num_procs: int = 4,
    ):
        super().__init__()
        self.datasets = datasets
        self.local_save_file = local_save_file
        self.config = config
        self.features = features
        self.labels = labels
        self.val_ratio = val_ratio
        self.data_collator = data_collator
        self.batch_size = batch_size
        self.seed = seed
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.num_procs = num_procs
        self.tokenize_before = tokenize_before

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.name_or_path)  # type: ignore

        if self.tokenize_before:
            self.data_collator = SingleLoaderCollator(
                self.features, self.labels, self.tokenizer.model_input_names
            )
        else:
            self.data_collator = SingleLoaderCollatorWithTokenize(
                self.features, self.labels, self.tokenizer, self.config
            )

        if isinstance(self.datasets, DatasetDict):
            logger.info(
                "we'll try to concatenate dataset, because we found it's a DatasetDict Type"
            )
            self.datasets = concatenate_datasets(
                [self.datasets[idx] for idx in self.datasets.keys()]
            )

    def prepare_data(self) -> None:
        if not self.tokenize_before:
            logger.info(f"since self.tokenize_before is {self.tokenize_before}, skipping prepare data")
            return 

        if os.path.exists(self.local_save_file):
            logger.info(
                f"cache file exist on {self.local_save_file}, loading cache file instead of preparing data"
            )

        datasets = self.datasets
        datasets = preprocess_tokenize_single_loader(
                datasets, self.config, self.features, self.num_procs, self.local_save_file #type:ignore
        )  

    def setup(self, stage: str) -> None:
        if self.tokenize_before:
            self.datasets = Dataset.load_from_disk(self.local_save_file)
            formatted_columns = (
                [
                    f"{feat}_{key}"
                    for feat in self.features
                    for key in self.tokenizer.model_input_names
                ]
                + self.labels
                if self.labels is not None
                else []
            )
            self.datasets.set_format("torch", columns=formatted_columns)
        else:
            formatted_columns = (
                self.features + self.labels if self.labels else self.features
            )
            self.datasets.set_format("torch", columns=formatted_columns)

        if stage == "fit":
            self.datasets = self.datasets.train_test_split(test_size=self.val_ratio, shuffle= self.shuffle) #type:ignore
            self.train_ds = self.datasets["train"]
            self.val_ds = self.datasets["test"]

    def train_dataloader(self):
        data_loader_params = {
            "num_workers": self.num_workers,
            "batch_size": self.batch_size,
            "shuffle": self.shuffle,
            "pin_memory": self.pin_memory,
            "persistent_workers": self.persistent_workers,
            "collate_fn": self.data_collator,
            "drop_last": self.drop_last
        }
        print("babi")
        return DataLoader(self.train_ds, **data_loader_params)  # type: ignore

    def val_dataloader(self):
        data_loader_params = {
            "num_workers": self.num_workers,
            "batch_size": self.batch_size,
            "shuffle": self.shuffle,
            "pin_memory": self.pin_memory,
            "persistent_workers": self.persistent_workers,
            "collate_fn": self.data_collator,
            "drop_last": self.drop_last
        }
        return DataLoader(self.val_ds, **data_loader_params)  # type: ignore
