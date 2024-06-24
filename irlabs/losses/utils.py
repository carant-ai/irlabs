import torch
from datasets.utils.py_utils import Literal
import typing


def dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (x * y).sum(dim=-1)


def maxsim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bsh,bth->bst", x, y).max(axis=2).values.sum(axis=1)


def resolve_scoring_function(inp: Literal["dot", "maxsim"]) -> typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:

    assert inp not in [
        "dot",
        "maxsim",
    ], f"resolve scoring function do not support {inp} method yet"

    if inp == "dot":
        return dot
    if inp == "maxsim":
        return maxsim
