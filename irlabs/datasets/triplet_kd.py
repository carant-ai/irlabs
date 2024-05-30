import enum
from torch.utils.data import Dataset
from tqdm import tqdm
from collections import defaultdict
from typing import List


class TripletKDDataset(Dataset):
    """pos_score, neg_score, anchor, pos, neg"""

    def __init__(
        self,
        queries_path: str,
        collection_path: str,
        quintuple_ids_path: str,
        labels: List[str] = ["positive_score", "negative_score"],
        seperator: str = "\t",
    ):
        self.queries = {}
        self.collection = {}
        self.quintuple_ids = []
        self.quintuple_mapping = []
        self.stats = defaultdict(int)
        with open(queries_path, "r", encoding="utf8") as f:
            for i, line in enumerate(tqdm(f)):
                if i == 0:
                    continue

                ls = line.split(seperator)
                self.queries[ls[0]] = ls[1]

        with open(collection_path, "r", encoding="utf8") as f:
            for i, line in enumerate(tqdm(f)):
                if i == 0:
                    continue

                ls = line.split(seperator)
                self.collection[ls[0]] = ls[1]

        with open(quintuple_ids_path, "r", encoding="utf8") as f:
            for i, line in enumerate(tqdm(f)):
                if i == 0:
                    ls = line.split(seperator)
                    self.quintuple_mapping = ls
                    continue

                ls = line.split(seperator)
                self.quintuple_ids.append(
                    (
                        float(ls[0]),
                        float(ls[1]),
                        ls[2],
                        ls[3],
                        ls[4],
                    )
                )

    def __getitem__(self, index):
        return {
            "positive_score": self.quintuple_ids[index][0],
            "negative_score": self.quintuple_ids[index][1],
            "anchor": self.queries[self.quintuple_ids[index][2]],
            "positive": self.collection[self.quintuple_ids[index][3]],
            "negative": self.collection[self.quintuple_ids[index][4]],
        }
