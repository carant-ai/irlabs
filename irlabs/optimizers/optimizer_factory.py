from typing import Dict, Any, Callable, Optional
from torch import nn
from torch import optim


def param_groups_weight_decay_default(model: nn.Module, weight_decay):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if param.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)

    return [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay, "weight_decay": weight_decay},
    ]


def optimizer_factory(
    model: nn.Module,
    optimizer_name: str,
    optimizer_hparams: Dict[str, Any],
):
    weight_decay = optimizer_hparams.pop("weight_decay", 0.0)
    parameters = param_groups_weight_decay_default(
        model,
        weight_decay,
    )

    if optimizer_name == "Adam":
        return optim.AdamW(parameters, **optimizer_hparams)
    elif optimizer_name == "SGD":
        return optim.SGD(parameters, **optimizer_hparams)
    else:
        assert (
            False
        ), f'Unknown optimizer name: "{optimizer_name}". Please choose either "Adam" or "SGD".'
