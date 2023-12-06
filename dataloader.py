import math
from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import torch
from torch.utils.data import Dataset

matplotlib.use("TkAgg")

T = TypeVar("T")


@dataclass
class DensitySample(Generic[T]):
    pdf: Callable[[T], float]
    samples: list[T]


class DensityDataset(Dataset):
    def __init__(self, samples: list):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index) -> torch.Tensor:
        return torch.tensor([self.samples[index]], dtype=torch.float32)


def get_1d_norm_sample(n_samples: int) -> DensitySample[float]:
    """

    Args:
        n_samples:

    Returns:

    """
    dist1 = scipy.stats.norm(0, 3)
    dist2 = scipy.stats.norm(10, 1)
    pdf: Callable[[float], float] = lambda x: dist1.pdf(x) * dist2.pdf(x)  # type: ignore
    samples: list[float] = list(dist1.rvs(size=n_samples // 2)) + list(dist2.rvs(size=n_samples // 2))  # type: ignore

    return DensitySample(pdf, samples)


def plot_1d_dataset(dataset: DensitySample[float]) -> None:
    """

    Args:
        dataset:
    """
    x_ticks = np.linspace(-5, 5, 1000)
    y_vals = [dataset.pdf(v) for v in x_ticks]

    plt.plot(x_ticks, y_vals)
    plt.scatter(dataset.samples, [0 for _ in dataset.samples])
    plt.show()


if __name__ == "__main__":
    dataset = get_1d_norm_sample(20)
    plot_1d_dataset(dataset)
