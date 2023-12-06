from dataclasses import dataclass
from enum import Enum

import torch
from torch import nn


class BasicSwish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class BasicModel(torch.nn.Module):
    def __init__(self, n_inputs: int):
        super().__init__()

        self.n_inputs = n_inputs
        self.net = torch.nn.Sequential(
            torch.nn.Linear(self.n_inputs, 3),
            torch.nn.Linear(3, 1),
            BasicSwish(),
        )

    def forward(self, x):
        return self.net(x)

    def score(self, x, sigma=None):
        x = x.requires_grad_()
        logp = -self.net(x).sum()
        return torch.autograd.grad(logp, x, create_graph=True)[0]


class ModelTypes(str, Enum):
    Basic = "basic"


@dataclass
class Config:
    model_type: ModelTypes = ModelTypes.Basic
    n_epochs: int = 10
    n_inputs: int = 2


MODEL_TYPE_MAPPING = {ModelTypes.Basic: BasicModel}


def load_model(config: Config):
    loaded_model_type = MODEL_TYPE_MAPPING[config.model_type]

    return loaded_model_type(
        n_inputs=config.n_inputs,
    )
