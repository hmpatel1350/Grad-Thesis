import torch

from torch.utils.data import DataLoader
import time
from matplotlib import pyplot as plt

from accelerate import Accelerator

# import tensor dataset and data loader
from torch.utils.data import TensorDataset, DataLoader

import wandb

import yaml

with open("06-rcp-train-on-2d_grid.yaml", "r") as file:
    config = yaml.safe_load(file)

# Project the data using PCA
def get_data_projector(X, n_components=2):
    X_reshape = X.reshape(-1, X.shape[-1])
    _, _, V = torch.svd(X_reshape)
    return (V.detach().cpu())[:, :n_components]

def read_data():
    paths_0 = []
    paths_1 = []
    for i in range(63):
        path_0 = torch.load(f"../Data/transformers_layers_0_input_output/transformer_layer_0_inputs_batch_{i}.pt")
        path_1 = torch.load(f"../Data/transformers_layers_0_input_output/transformer_layer_0_outputs_batch_{i}.pt")
        paths_0.append(path_0)
        paths_1.append(path_1)

    paths_0 = torch.cat(paths_0).detach().cpu()
    paths_1 = torch.cat(paths_1).detach().cpu()
    return paths_0, paths_1


def plot_path(path, c='r', label=None):
    # get the vectors
    vectors = path[1:] - path[:-1]
    # get the start points of the vectors
    start_points = path[:-1]
    # plot the path
    plt.quiver(*start_points.T, *vectors.T, scale_units='xy', angles='xy', scale=1, color=c, label=label)

paths_0 = {}
paths_1 = {}

paths_0['orig'], paths_1['orig'] = read_data()

paths_0['100d_projector'] = get_data_projector(paths_0['orig'], n_components=100)
paths_0['100d'] = paths_0['orig'] @ paths_0['100d_projector']

paths_1['100d_projector'] = get_data_projector(paths_1['orig'], n_components=100)
paths_1['100d'] = paths_1['orig'] @ paths_1['100d_projector']

paths_0['2d_projector'] = get_data_projector(paths_0['orig'], n_components=2)
paths_0['2d'] = paths_0['orig'] @ paths_0['2d_projector']

paths_1['2d_projector'] = get_data_projector(paths_1['orig'], n_components=2)
paths_1['2d'] = paths_1['orig'] @ paths_1['2d_projector']


run = wandb.init(
    # Set the project where this run will be logged
    project="Graduate-Thesis",
    name = "AutoencoderOriginal2DGrid"
)
dataset = TensorDataset(paths_0['orig'], paths_1['orig'])
dataloader = DataLoader(dataset, batch_size=64)

iterations = 4
data_dimension = paths_0['orig'].shape[-1]
path_length = paths_0['orig'].shape[1]
h_size = 64
hidden_size = 128
print(path_length)
# Run an GRU on the data
class GridModel(torch.nn.Module):
    def __init__(self, data_dimension, h_size=64, hidden_size=128):
        super().__init__()
        self.data_dimension = data_dimension
        self.h_size = h_size
        self.hidden_size = hidden_size
        self.model = torch.nn.Sequential(torch.nn.Linear(in_features=self.data_dimension+2*self.h_size, out_features=self.hidden_size),
                                         torch.nn.GELU(),
                                         torch.nn.Linear(in_features=self.hidden_size, out_features=self.data_dimension+self.h_size))

    def forward(self, P0, h_i, h_j):
        z = torch.cat([P0, h_i, h_j], dim=-1)
        out = self.model(z)
        return out[:, :self.data_dimension], out[:, self.data_dimension:]


import torch.nn as nn
class GRUModel(nn.Module):
    def __init__(self, data_dimension, h_size, num_layers=4):
        super(GRUModel, self).__init__()
        self.data_dimension = data_dimension
        self.h_size = h_size
        self.num_layers = num_layers
        self.gru = nn.GRU(data_dimension, 2*self.h_size, num_layers, batch_first=True)
        self.fc = nn.Linear(2*self.h_size, data_dimension+self.h_size)

    def forward(self, P0, hi, hj):
        h0 = torch.cat([hi, hj], dim=-1)
        h0 = torch.tile(h0, (self.num_layers, 1, 1))
        out, _ = self.gru(P0, h0)
        out = self.fc(out[:, -1, :])
        return out[:, :self.data_dimension], out[:, self.data_dimension:]


class Autoencoder(nn.Module):
    def __init__(self, data_dimension, h_size):
        super(Autoencoder, self).__init__()
        self.data_dimension = data_dimension
        self.h_size = h_size
        self.model = torch.nn.Sequential(torch.nn.Linear(in_features=self.data_dimension+2*self.h_size, out_features=1024),
                                         torch.nn.GELU(),
                                         torch.nn.Linear(in_features=1024,
                                                         out_features=512),
                                         torch.nn.ReLU(),
                                         torch.nn.Linear(in_features=512,
                                                         out_features=1024),
                                         torch.nn.GELU(),
                                         torch.nn.Linear(in_features=1024, out_features=self.data_dimension+self.h_size))
    def forward(self, P0, h_i, h_j):
        z = torch.cat([P0, h_i, h_j], dim=-1)
        out = self.model(z)
        return out[:, :self.data_dimension], out[:, self.data_dimension:]

model = Autoencoder(data_dimension, h_size)
model = GridModel(data_dimension, h_size, hidden_size)
# Train the model
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss()

# use the accelerator for distributed training
accelerator = Accelerator()
model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
start_time = time.time()
only_gpu = 0.0
for epoch in range(config['epochs']):
    for (P0, P1) in dataloader:
        optimizer.zero_grad()
        loss = 0.0
        # a 2 dimension list of m by n of None values
        # this will be used to store the hidden states
        H = [[None for _ in range(path_length)] for _ in range(iterations)]
        for j in range(path_length):
            for i in range(iterations):
                if i == 0:
                    h_i = torch.zeros(P0.shape[0], h_size, device=accelerator.device)
                else:
                    h_i = H[i-1][j]
                if j == 0:
                    h_j = torch.zeros(P0.shape[0], h_size, device=accelerator.device)
                else:
                    h_j = H[i][j-1]
                start_gpu = time.time()
                out, h = model(P0[:, j], h_i, h_j)
                end_gpu = time.time()
                only_gpu += (end_gpu - start_gpu)
                H[i][j] = h
                loss += loss_fn(out, P1[:, j])

        wandb.log({'loss': loss})
        loss.backward()
        optimizer.step()
print("Test1")

end_time = time.time()
elapsed_time = end_time - start_time

P0 = paths_0['orig'].to(accelerator.device)
P1 = torch.zeros_like(P0, device=accelerator.device)
H = [[None for _ in range(path_length)] for _ in range(iterations)]
for j in range(path_length):
    for i in range(iterations):
        if i == 0:
            h_i = torch.zeros(P0.shape[0], h_size, device=accelerator.device)
        else:
            h_i = H[i-1][j]
        if j == 0:
            h_j = torch.zeros(P0.shape[0], h_size, device=accelerator.device)
        else:
            h_j = H[i][j-1]
        P1[:, j], h = model(P0[:, j], h_i, h_j)
        H[i][j] = h
print("Test2")

paths_1['fast_grid_pred'] = P1.detach().cpu()
paths_1['fast_grid_pred_2d'] = paths_1['fast_grid_pred'] @ paths_1['2d_projector']

plot_path(paths_1['2d'][0], c='r')
plot_path(paths_1['fast_grid_pred_2d'][0], c='y')
plt.suptitle("Oringal path (red) and Grid prediction (yellow)")
plt.title(f'Time to Train: {elapsed_time}s Time Running Model: {only_gpu}s')
plt.savefig("paths_comparison.png")
wandb.log({"paths_comparison": wandb.Image("paths_comparison.png")})
print("Test3")
