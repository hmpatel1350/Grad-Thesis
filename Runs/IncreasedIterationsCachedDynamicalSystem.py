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
        path_0 = torch.load(f"J:/PyCharmData/inn_sequence/data/2024-09-07_18-04-35/transformer_layer_0_inputs_batch_{i}.pt")
        path_1 = torch.load(f"J:/PyCharmData/inn_sequence/data/2024-09-07_18-04-35/transformer_layer_0_outputs_batch_{i}.pt")
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
    name = "Increased Iterations"
)

dataset = TensorDataset(paths_0['orig'], paths_1['orig'])
# NOTE: The shuffle=False is important!  Since we want to keeps the paths together and ordered
# in the dataload to keep the H consistent
dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

iterations_max = 4
data_dimension = paths_0['orig'].shape[-1]
path_length = paths_0['orig'].shape[1]
h_size = 64
hidden_size = 128

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

model = GridModel(data_dimension, h_size, hidden_size)
# Train the model
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss()

# use the accelerator for distributed training
accelerator = Accelerator()
model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
start_time = time.time()
only_gpu = 0.0

H_cache = []

# We pad H with zeros when i or j are out of bounds
def getH(batch_idx):
    return H_cache[batch_idx]

for epoch in range(config['epochs']):
    iterations = 1 + (epoch * iterations_max) // (config['epochs'])

    # Precompute for the current epoch
    for batch_idx, (P0, P1) in enumerate(dataloader):
        P0_repeated = P0.unsqueeze(1).repeat(1, iterations, 1, 1)
        P1_repeated = P1.unsqueeze(1).repeat(1, iterations, 1, 1)

        optimizer.zero_grad()
        loss = 0.0
        # a 2 dimension list of m by n of None values
        # this will be used to store the hidden states
        if epoch == 0:
            H_initial = torch.zeros(P0.shape[0], iterations, path_length, h_size, device=accelerator.device)
            H_cache.append(H_initial)

        H_old = getH(batch_idx)


        zeros_i = torch.zeros(H_old.shape[0], 1, H_old.shape[2], H_old.shape[3], device=accelerator.device)
        H_i = torch.cat((zeros_i, H_old), dim=1)
        if(epoch % 25 != 0 or epoch == 0):
            H_i = H_i[:, :-1, :, :]

        zeros_j = torch.zeros(H_old.shape[0], H_old.shape[1], 1, H_old.shape[3], device=accelerator.device)
        H_j = torch.cat((zeros_j, H_old), dim=2)
        H_j = H_j[:, :, :-1, :]
        if(epoch % 25 == 0 and epoch != 0):
            H_j = torch.cat((H_j, zeros_i), dim=1)

        P0_reshaped = P0_repeated.reshape(-1, data_dimension)
        H_i_reshaped = H_i.reshape(-1, h_size)
        H_j_reshaped = H_j.reshape(-1, h_size)

        start_gpu = time.time()
        out, h = model(P0_reshaped, H_i_reshaped, H_j_reshaped)
        end_gpu = time.time()
        only_gpu += (end_gpu - start_gpu)
        out_reshaped = out.reshape(P0_repeated.shape[0], P0_repeated.shape[1], P0_repeated.shape[2],
                                   P0_repeated.shape[3])
        loss = 0.0

        for i in range(iterations):
            loss += (i+1) * loss_fn(out_reshaped[:,i,:,:], P1_repeated[:,i,:,:])

        wandb.log({'loss': loss})
        loss.backward()
        optimizer.step()

        H_cache[batch_idx] = h.reshape(P0.shape[0], iterations, path_length, h_size).detach()

print("Test1")

end_time = time.time()
elapsed_time = end_time - start_time

P0 = paths_0['orig'].to(accelerator.device)
P1 = torch.zeros_like(P0, device=accelerator.device)
H = [[None for _ in range(path_length)] for _ in range(iterations_max)]
for j in range(path_length):
    for i in range(iterations_max):
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
