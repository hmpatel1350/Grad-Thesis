import time

import torch
from accelerate import Accelerator
from huggingface_hub._webhooks_payload import BaseModel
from matplotlib import pyplot as plt
import wandb

class BaseModelInterface:
    def __init__(self, name, model, wandb_run, config, dataloader, data_dimension, optimizer, loss_fn, path_length=None,
                 iterations=None, h_size=None, hidden_size=None, num_layers=None, dropout=None):
        self.name = name

        self.data_dimension = data_dimension
        self.h_size = h_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.config = config
        self.path_length = path_length
        self.iterations = iterations
        self.wandb_run = wandb_run

        self.accelerator = Accelerator()
        self.model, self.optimizer, self.dataloader = self.accelerator.prepare(model, optimizer, dataloader)

    def train(self):
        only_gpu = 0.0
        start_time = time.time()
        for epoch in range(self.config['epochs']):
            for i, (start, end, attention_mask) in enumerate(self.dataloader):
                self.optimizer.zero_grad()

                start_gpu = time.time()
                out = self.model(start)
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

    def eval(self, paths_0, paths_1):
        model = self.model.to('cpu')
        paths_1[f'{self.name}_pred'] = model(paths_0['orig']).detach()
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

    def plot_path(self, path, c='r', label=None):
        # get the vectors
        vectors = path[1:] - path[:-1]
        # get the start points of the vectors
        start_points = path[:-1]
        # plot the path
        plt.quiver(*start_points.T, *vectors.T, scale_units='xy', angles='xy', scale=1, color=c, label=label)


class Model(torch.nn.Module):
    def __init__(self, data_dimension=None, h_size=None, hidden_size=None, num_layers=None, dropout=None):
        super().__init__()
        self.data_dimension = data_dimension
        self.h_size = h_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.linear = torch.nn.Linear(self.data_dimension, self.data_dimension)

    def forward(self, P0=None, Hi=None, Hj=None):
        out = self.linear(P0)
        return out
