import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from dataloader import DensityDataset, DensitySample, get_1d_norm_sample, sample_2d
from models import Config, Energy, ModelTypes, ToyMLP, load_model
from score_matching import exact_1d_score_matching, ssm_loss


class Trainer:
    def __init__(self, config: Config, dataset: DensitySample):
        self.config = config
        self.data = dataset

        # self.dataset = DensityDataset(self.data.samples)
        # self.dataloader = DataLoader(
        #     self.dataset,
        #     batch_size=1,
        # )
        self.model = load_model(config)
        self.dataloader = DataLoader(sample_2d("8gaussians", 250).numpy())
        # self.model = Energy(ToyMLP())

    def train(self):
        optimizer = AdamW(self.model.parameters())
        for epoch_idx in range(self.config.n_epochs):
            print(epoch_idx)
            for x in self.dataloader:
                x.requires_grad_(True)
                # p = self.model.forward(x)
                # loss = exact_1d_score_matching(p, x)
                loss = ssm_loss(self.model, x, torch.randn_like(x))
                print(loss)

                loss.backward()
                optimizer.step()

                optimizer.zero_grad()


if __name__ == "__main__":
    config = Config(
        model_type=ModelTypes.Basic,
        n_epochs=10,
    )

    dataset = get_1d_norm_sample(250)

    trainer = Trainer(
        config=config,
        dataset=dataset,
    )

    trainer.train()
