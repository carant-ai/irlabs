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
from .collator import SingleLoaderCollatorWithTokenize

logger = logging.getLogger(__name__)


class HFSingleLoaderModule(LightningDataModule):
    def __init__(
        self,
        datasets: Dataset | DatasetDict,
        ir_config: PretrainedConfig,
        query_columns: List[str],
        document_columns: List[str],
        labels: List[str] | None, #used for knowledge distill
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
        self.ir_config = ir_config
        self.query_columns = query_columns
        self.document_columns = document_columns
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

        self.tokenizer = AutoTokenizer.from_pretrained(self.ir_config.name_or_path)  # type: ignore

        self.data_collator = SingleLoaderCollatorWithTokenize(
            self.query_columns,self.document_columns, self.labels, self.tokenizer, self.ir_config
        )

        if isinstance(self.datasets, DatasetDict):
            logger.info(
                "we'll try to concatenate dataset, because we found it's a DatasetDict Type"
            )
            self.datasets = concatenate_datasets(
                [self.datasets[idx] for idx in self.datasets.keys()]
            )

    def prepare_data(self) -> None:
        return

    def setup(self, stage: str) -> None:
        formatted_columns = (
            self.query_columns+ self.document_columns 
        )
        formatted_columns += self.labels if self.labels is not None else []
        self.datasets.set_format("torch", columns=formatted_columns)

        assert isinstance(self.datasets, Dataset)

        if stage == "fit":
            self.datasets = self.datasets.train_test_split(
                test_size=self.val_ratio, shuffle=self.shuffle
            )  # type:ignore
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
            "drop_last": self.drop_last,
        }
        return DataLoader(self.train_ds, **data_loader_params)  # type: ignore

    def val_dataloader(self):
        data_loader_params = {
            "num_workers": self.num_workers,
            "batch_size": self.batch_size,
            "shuffle": self.shuffle,
            "pin_memory": self.pin_memory,
            "persistent_workers": self.persistent_workers,
            "collate_fn": self.data_collator,
            "drop_last": self.drop_last,
        }
        return DataLoader(self.val_ds, **data_loader_params)  # type: ignore
