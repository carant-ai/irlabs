from lightning import LightningDataModule
from torch.utils.data import DataLoader

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
from .collator import SingleLoaderCollator

logger = logging.getLogger(__name__)


class SingleLoaderModule(LightningDataModule):
    def __init__(
        self,
        datasets: Dataset | DatasetDict,
        local_save_file: str,
        config: Optional[PretrainedConfig],
        features: List[str],
        labels: List[str] | None,
        val_ratio: float | None,
        data_collator: Optional[Any] = None,
        batch_size: int = 4,
        seed: int = 42,
        drop_last: bool = False,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = False,
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

    def prepare_data(self) -> None:
        datasets = self.datasets

        if self.config is None:
            logger.warning("config params must present")
            raise ValueError("config params must present")

        if isinstance(datasets, DatasetDict):
            logger.info(
                "we'll try to concatenate dataset, because we found it's a DatasetDict Type"
            )
            datasets = concatenate_datasets([datasets[idx] for idx in datasets.keys()])

        datasets = preprocess_tokenize_single_loader(
            datasets, self.config, self.features, self.num_workers
        )
        datasets.save_to_disk(self.local_save_file)

    def setup(self, stage: str) -> None:
        self.datasets = Dataset.load_from_disk(self.local_save_file)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.name_or_path)  # type: ignore
        formatted_columns = [
            f"{feat}_{key}"
            for feat in self.features
            for key in self.tokenizer.model_input_names
        ] + self.labels if self.labels is not None else []
        self.datasets.set_format("torch", columns=formatted_columns)
        if stage == "fit":
            self.datasets = self.datasets.train_test_split(test_size=self.val_ratio)
            self.train_ds = self.datasets["train"]
            self.val_ds = self.datasets["test"]

    def train_dataloader(self):
        data_collator = SingleLoaderCollator(
            self.features, self.labels, self.tokenizer.model_input_names
        )
        data_loader_params = {
            "num_workers": self.num_workers,
            "batch_size": self.batch_size,
            "pin_memory": self.pin_memory,
            "persistent_workers": self.persistent_workers,
            "collate_fn": data_collator,
        }
        return DataLoader(self.train_ds, **data_loader_params)  # type: ignore

    def val_dataloader(self):
        data_collator = SingleLoaderCollator(
            self.features, self.labels, self.tokenizer.model_input_names
        )
        data_loader_params = {
            "num_workers": self.num_workers,
            "batch_size": self.batch_size,
            "pin_memory": self.pin_memory,
            "persistent_workers": self.persistent_workers,
            "collate_fn": data_collator,
        }
        return DataLoader(self.val_ds, **data_loader_params)  # type: ignore
