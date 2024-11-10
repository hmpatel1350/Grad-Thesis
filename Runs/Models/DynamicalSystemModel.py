import time

import torch
import wandb
from matplotlib import pyplot as plt

from Models.BaseModelInterface import BaseModelInterface


class DynamicalSystemModel(BaseModelInterface):
    def train(self):
        only_gpu = 0.0
        start_time = time.time()
        for epoch in range(self.config['epochs']):
            for (P0, P1, attention_mask) in self.dataloader:
                self.optimizer.zero_grad()
                loss = 0.0
                # a 2 dimension list of m by n of None values
                # this will be used to store the hidden states
                H = [[None for _ in range(self.path_length)] for _ in range(self.iterations)]
                for j in range(self.path_length):
                    for i in range(self.iterations):
                        if i == 0:
                            h_i = torch.zeros(P0.shape[0], self.h_size, device=self.accelerator.device)
                        else:
                            h_i = H[i - 1][j]
                        if j == 0:
                            h_j = torch.zeros(P0.shape[0], self.h_size, device=self.accelerator.device)
                        else:
                            h_j = H[i][j - 1]

                        start_gpu = time.time()
                        out, h = self.model(P0[:, j], h_i, h_j)
                        end_gpu = time.time()
                        only_gpu += (end_gpu - start_gpu)

                        H[i][j] = h
                        loss += self.loss_fn(out, P1[:, j])
                self.wandb_run.log({f'{self.name}_loss': loss})
                loss.backward()
                self.optimizer.step()
        end_time = time.time()
        elapsed_time = end_time - start_time
        self.elapsed_time = elapsed_time
        self.only_gpu = only_gpu
        return elapsed_time, only_gpu

    def eval(self, paths_0, paths_1):
        P0 = paths_0['orig'].to(self.accelerator.device)
        P1 = torch.zeros_like(P0, device=self.accelerator.device)
        H = [[None for _ in range(self.path_length)] for _ in range(self.iterations)]
        for j in range(self.path_length):
            for i in range(self.iterations):
                if i == 0:
                    h_i = torch.zeros(P0.shape[0], self.h_size, device=self.accelerator.device)
                else:
                    h_i = H[i - 1][j]
                if j == 0:
                    h_j = torch.zeros(P0.shape[0], self.h_size, device=self.accelerator.device)
                else:
                    h_j = H[i][j - 1]
                P1[:, j], h = self.model(P0[:, j], h_i, h_j)
                H[i][j] = h

        paths_1[f'{self.name}_pred'] = P1.detach().cpu()
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
