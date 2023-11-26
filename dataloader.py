from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

import numpy as np
import scipy.stats

T = TypeVar("T")


@dataclass
class DensitySample(Generic[T]):
    pdf: Callable[[T], float]
    samples: list[T]


def get_1d_norm_sample(n_samples: int) -> DensitySample[float]:
    dist = scipy.stats.norm
    pdf: Callable[[float], float] = dist.pdf  # type: ignore
    samples: list[float] = dist.rvs(size=n_samples)  # type: ignore

    return DensitySample(pdf, samples)
