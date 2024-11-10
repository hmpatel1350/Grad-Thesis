import time

import torch
import wandb
from matplotlib import pyplot as plt

from Models.BaseModelInterface import BaseModelInterface

class FalconLayer0Model(BaseModelInterface):
    def train(self):
        only_gpu = 0.0
        start_time = time.time()
        for epoch in range(self.config['epochs']):
            for i, (start, end, attention_mask) in enumerate(self.dataloader):
                self.optimizer.zero_grad()

                start_gpu = time.time()
                out = self.model(start, attention_mask=attention_mask, alibi=None)[0]
                end_gpu = time.time()
                only_gpu += (end_gpu - start_gpu)

                loss = self.loss_fn(out, end)
                loss.backward()
                self.wandb_run.log({f'{self.name}_loss': loss})
                self.optimizer.step()
        end_time = time.time()
        elapsed_time = end_time - start_time
        self.elapsed_time = elapsed_time
        self.only_gpu = only_gpu
        return elapsed_time, only_gpu
    def eval(self, paths_0, paths_1, attention_masks):
        model = self.model.to('cpu')
        paths_1[f'{self.name}_pred'] = model(paths_0['orig'][0:64], attention_mask=attention_masks[0:64], alibi=None)[0].detach()
        paths_1[f'{self.name}_pred_2d'] = paths_1[f'{self.name}_pred'] @ paths_1['2d_projector']
        self.plot_path(paths_1['2d'][0], c='r')
        self.plot_path(paths_1[f'{self.name}_pred_2d'][0], c='b')
        plt.suptitle(f'Original Path (red) and {self.name} Prediction (blue)')
        plt.title(f'Train Time: {self.elapsed_time}s Only GPU: {self.only_gpu}s')
        plt.savefig("paths_comparison.png")
        self.wandb_run.log({f'{self.name}_comparison': wandb.Image("paths_comparison.png")})
        plt.clf()
        plt.cla()
        plt.close()
        return paths_1[f'{self.name}_pred'], paths_1[f'{self.name}_pred_2d']

class GridModel(torch.nn.Module):
    def __init__(self, data_dimension, h_size=64, hidden_size=128):
        super().__init__()
        self.data_dimension = data_dimension
        self.h_size = h_size
        self.hidden_size = hidden_size
        self.model = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.data_dimension + 2 * self.h_size, out_features=1024),
            torch.nn.GELU(),
            torch.nn.Linear(in_features=1024,
                            out_features=512),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=512,
                            out_features=1024),
            torch.nn.GELU(),
            torch.nn.Linear(in_features=1024, out_features=self.data_dimension + self.h_size))

    def forward(self, P0, h_i, h_j):
        z = torch.cat([P0, h_i, h_j], dim=-1)
        out = self.model(z)
        return out[:, :self.data_dimension], out[:, self.data_dimension:]

