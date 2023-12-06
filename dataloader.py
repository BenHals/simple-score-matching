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
    dist = scipy.stats.norm
    pdf: Callable[[float], float] = dist.pdf  # type: ignore
    samples: list[float] = dist.rvs(size=n_samples)  # type: ignore

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


def sample_2d(dataset, n_samples):
    z = torch.randn(n_samples, 2)

    if dataset == "8gaussians":
        scale = 4
        sq2 = 1 / math.sqrt(2)
        centers = [
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (sq2, sq2),
            (-sq2, sq2),
            (sq2, -sq2),
            (-sq2, -sq2),
        ]
        centers = torch.tensor([(scale * x, scale * y) for x, y in centers])
        return sq2 * (0.5 * z + centers[torch.randint(len(centers), size=(n_samples,))])

    elif dataset == "2spirals":
        n = torch.sqrt(torch.rand(n_samples // 2)) * 540 * (2 * math.pi) / 360
        d1x = -torch.cos(n) * n + torch.rand(n_samples // 2) * 0.5
        d1y = torch.sin(n) * n + torch.rand(n_samples // 2) * 0.5
        x = (
            torch.cat(
                [torch.stack([d1x, d1y], dim=1), torch.stack([-d1x, -d1y], dim=1)],
                dim=0,
            )
            / 3
        )
        return x + 0.1 * z

    elif dataset == "checkerboard":
        x1 = torch.rand(n_samples) * 4 - 2
        x2_ = (
            torch.rand(n_samples)
            - torch.randint(0, 2, (n_samples,), dtype=torch.float) * 2
        )
        x2 = x2_ + x1.floor() % 2
        return torch.stack([x1, x2], dim=1) * 2

    elif dataset == "rings":
        n_samples4 = n_samples3 = n_samples2 = n_samples // 4
        n_samples1 = n_samples - n_samples4 - n_samples3 - n_samples2

        # so as not to have the first point = last point, set endpoint=False in np; here shifted by one
        linspace4 = torch.linspace(0, 2 * math.pi, n_samples4 + 1)[:-1]
        linspace3 = torch.linspace(0, 2 * math.pi, n_samples3 + 1)[:-1]
        linspace2 = torch.linspace(0, 2 * math.pi, n_samples2 + 1)[:-1]
        linspace1 = torch.linspace(0, 2 * math.pi, n_samples1 + 1)[:-1]

        circ4_x = torch.cos(linspace4)
        circ4_y = torch.sin(linspace4)
        circ3_x = torch.cos(linspace4) * 0.75
        circ3_y = torch.sin(linspace3) * 0.75
        circ2_x = torch.cos(linspace2) * 0.5
        circ2_y = torch.sin(linspace2) * 0.5
        circ1_x = torch.cos(linspace1) * 0.25
        circ1_y = torch.sin(linspace1) * 0.25

        x = (
            torch.stack(
                [
                    torch.cat([circ4_x, circ3_x, circ2_x, circ1_x]),
                    torch.cat([circ4_y, circ3_y, circ2_y, circ1_y]),
                ],
                dim=1,
            )
            * 3.0
        )

        # random sample
        x = x[torch.randint(0, n_samples, size=(n_samples,))]

        # Add noise
        return x + torch.normal(mean=torch.zeros_like(x), std=0.08 * torch.ones_like(x))

    else:
        raise RuntimeError("Invalid `dataset` to sample from.")


if __name__ == "__main__":
    dataset = get_1d_norm_sample(20)
    plot_1d_dataset(dataset)
