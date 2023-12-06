import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from dataloader import DensityDataset, DensitySample, get_1d_norm_sample
from models import Config, ModelTypes, load_model
from score_matching import exact_1d_score_matching, ssm_loss


def plot_model_predictions(
    model: torch.nn.Module, dataset: DensitySample[float]
) -> None:
    """

    Args:
        dataset:
    """
    x_ticks = np.linspace(-20, 20, 1000)
    x = torch.concat([torch.tensor([v], dtype=torch.float32) for v in x_ticks])
    with torch.inference_mode():
        y_vals = model.forward(x.unsqueeze(1)).cpu().detach()
        y_vals = torch.exp(y_vals)

    plt.plot(x_ticks, y_vals)
    plt.scatter(dataset.samples, [0 for _ in dataset.samples])
    plt.show()


class Trainer:
    def __init__(self, config: Config, dataset: DensitySample):
        self.config = config
        self.data = dataset

        self.dataset = DensityDataset(self.data.samples)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=1,
        )
        self.model = load_model(config)

    def train(self):
        optimizer = AdamW(self.model.parameters())
        for epoch_idx in range(self.config.n_epochs):
            print(epoch_idx)
            for x in self.dataloader:
                x.requires_grad_(True)
                # scores = self.model.score(x)
                # loss = exact_1d_score_matching(scores, x)
                loss = ssm_loss(self.model, x, torch.randn_like(x))
                print(loss)

                loss.backward()
                optimizer.step()

                optimizer.zero_grad()

        plot_model_predictions(self.model, self.data)


if __name__ == "__main__":
    config = Config(model_type=ModelTypes.Basic, n_epochs=5000, n_inputs=1)

    dataset = get_1d_norm_sample(250)

    trainer = Trainer(
        config=config,
        dataset=dataset,
    )

    trainer.train()
