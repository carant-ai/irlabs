from multiprocessing import Value
from irlabs.models.config import IRLabsConfig
from lightning import LightningDataModule
from torch.utils.data import DataLoader

from typing import Dict, List, Any, Optional
from datasets import (
    IterableDatasetDict,
    load_dataset,
    DatasetDict,
    Dataset,
    concatenate_datasets,
    load_from_disk,
    IterableDataset,
)
import logging
from dataclasses import dataclass

from transformers import AutoTokenizer 
from .utils import preprocess_tokenize_single_loader
from .collator import SingleLoaderCollator

logger = logging.getLogger(__name__)


@dataclass
class SentenceSingleLoaderModule(LightningDataModule):
    hf_load_dataset_args: Dict[str, Any]
    local_save_file: str
    config: Optional[IRLabsConfig] 
    features: List[str]
    labels: List[str] | None
    val_ratio: float | None
    data_collator: Optional[Any] = None
    batch_size: int = 32
    seed: int = 42
    drop_last: bool = False
    shuffle: bool = True
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = False

    def prepare_data(self) -> None:
        self.train_ds = load_dataset(**self.hf_load_dataset_args)
        self.data_loader_params = {}
        if self.config is None:
            logger.warning("config params must present")
            raise ValueError("config params must present")

        if isinstance(self.train_ds, IterableDataset):
            logger.warning("we haven't yet supported IterableDataset Class")
            raise ValueError("we haven't yet supported IterableDataset Class")

        if isinstance(self.train_ds, DatasetDict) or isinstance(
            self.train_ds, IterableDatasetDict
        ):
            logger.info(
                "we'll try to concatenate dataset, because we found it's a DatasetDict Type"
            )
            self.train_ds = concatenate_datasets(
                [self.train_ds[idx] for idx in self.train_ds.keys()]
            )

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.name_or_path)
        self.train_ds = preprocess_tokenize_single_loader(
            self.train_ds, self.config, self.features
        )
        self.train_ds.save_to_disk(self.local_save_file)

    def setup(self, stage: str) -> None:
        self.ds = Dataset.load_from_disk(self.local_save_file)
        if stage == "fit":
            self.ds = self.ds.train_test_split(test_size=self.val_ratio)
            self.train_ds = self.ds["train"]
            self.val_ds = self.ds["val"]

    def train_dataloader(self):
        data_collator = SingleLoaderCollator(self.features, self.labels, self.tokenizer.model_input_names)
        data_loader_params = {
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "persistent_workers": self.persistent_workers,
            "collate_fn": data_collator
        }
        return DataLoader(self.train_ds, **data_loader_params) 

    def val_dataloader(self):
        return DataLoader(self.val_ds, **self.data_loader_params)
