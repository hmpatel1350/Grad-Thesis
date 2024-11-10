import torch

from matplotlib import pyplot as plt

# import tensor dataset and data loader
from torch.utils.data import TensorDataset, DataLoader

import wandb

import yaml

from Models.BaseModelInterface import BaseModelInterface
from Models.FalconLayer0Model import FalconLayer0Model
from Models.LSTMModel import LSTMModel
from Models.GRUModel import GRUModel
from Models.DynamicalSystemModel import GridModel
from Models.DynamicalSystemModel import DynamicalSystemModel
from Models.CachedDynamicalModel import CachedDynamicalSystemModel
from Models.IncreasediterationslModel import IncreasedIterationsModel


from transformers.models.falcon.modeling_falcon import FalconDecoderLayer
from transformers.models.falcon.configuration_falcon import FalconConfig
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
    attention_masks = []
    for i in range(63):
        path_0 = torch.load(f"../Data/transformers_layers_0_input_output/transformer_layer_0_inputs_batch_{i}.pt")
        path_1 = torch.load(f"../Data/transformers_layers_0_input_output/transformer_layer_0_outputs_batch_{i}.pt")
        attention_mask = torch.load(f"../Data/attention_mask/attention_mask_batch_{i}.pt")
        paths_0.append(path_0)
        paths_1.append(path_1)
        attention_masks.append(attention_mask)
    paths_0 = torch.cat(paths_0).detach().cpu()
    paths_1 = torch.cat(paths_1).detach().cpu()
    attention_masks = torch.cat(attention_masks).detach().cpu()
    return paths_0[0:960], paths_1[0:960], attention_masks[0:960]


def plot_path(path, c='r', label=None):
    # get the vectors
    vectors = path[1:] - path[:-1]
    # get the start points of the vectors
    start_points = path[:-1]
    # plot the path
    plt.quiver(*start_points.T, *vectors.T, scale_units='xy', angles='xy', scale=1, color=c, label=label)


paths_0 = {}
paths_1 = {}

paths_0['orig'], paths_1['orig'], attention_masks_data = read_data()

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
    name="All Models"
)

dataset = TensorDataset(paths_0['orig'], paths_1['orig'], attention_masks_data)
# NOTE: The shuffle=False is important!  Since we want to keeps the paths together and ordered
# in the dataload to keep the H consistent
dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

iterations = 4
data_dimension = paths_0['orig'].shape[-1]
path_length = paths_0['orig'].shape[1]
h_size = 64
hidden_size = 128
num_layers = 4
dropout = 0.5
# Train the model
loss_fn = torch.nn.MSELoss()


########FalconLayer0##########
falconConfig = FalconConfig(hidden_size=2048, ffn_hidden_size=8192, num_attention_heads=1, parallel_attn=False, alibi=False)
falconLayer = FalconDecoderLayer(config=falconConfig)
print(f'falconLayer\n{falconLayer}')
falconLayer0Model = torch.load("./Models/transformer_layer_0.pt")
print(f'falconLayer0Model\n{falconLayer0Model}')
falconLayer0Optimizer = torch.optim.Adam(falconLayer.parameters(), lr=1e-4)

falconLayer0Training = FalconLayer0Model("FalconLayer0", falconLayer, run, config, dataloader, data_dimension,
                                  falconLayer0Optimizer, loss_fn, path_length=path_length, hidden_size=hidden_size,
                                  iterations=iterations, h_size=h_size, num_layers=num_layers, dropout=dropout)

elapsed_time, only_gpu = falconLayer0Training.train()
print(f'FalconLayer0 - Elapsed Time: {elapsed_time}, Only GPU: {only_gpu}')
falconLayer0Pred, falconLayer0Pred2D = falconLayer0Training.eval(paths_0, paths_1, attention_masks_data)

########LSTM##########
lstmModel = LSTMModel(data_dimension)
lstm_optimizer = torch.optim.Adam(lstmModel.parameters(), lr=1e-3)

lstmTraining = BaseModelInterface("LSTM", lstmModel, run, config, dataloader, data_dimension,
                                  lstm_optimizer, loss_fn, path_length=path_length, hidden_size=hidden_size,
                                  iterations=iterations, h_size=h_size, num_layers=num_layers, dropout=dropout)
elapsed_time, only_gpu = lstmTraining.train()
print(f'LSTM - Elapsed Time: {elapsed_time}, Only GPU: {only_gpu}')
lstmPred, lstmPred2D = lstmTraining.eval(paths_0, paths_1)

########GRU##########
gruModel = GRUModel(data_dimension)
gru_optimizer = torch.optim.Adam(gruModel.parameters(), lr=1e-3)

gruTraining = BaseModelInterface("GRU", gruModel, run, config, dataloader, data_dimension,
                                 gru_optimizer, loss_fn, path_length=path_length, hidden_size=hidden_size,
                                 iterations=iterations, h_size=h_size, num_layers=num_layers, dropout=dropout)
elapsed_time, only_gpu = gruTraining.train()
print(f'GRU - Elapsed Time: {elapsed_time}, Only GPU: {only_gpu}')
gruPred, gruPred2D = gruTraining.eval(paths_0, paths_1)

########DynamicalBase##########
baseDynamicalModel = GridModel(data_dimension)
baseDynamicalOptimizer = torch.optim.Adam(baseDynamicalModel.parameters(), lr=1e-4)

baseDynamicalTraining = DynamicalSystemModel("BaseDynamical", baseDynamicalModel, run, config, dataloader,
                                             data_dimension,
                                             baseDynamicalOptimizer, loss_fn, path_length=path_length,
                                             hidden_size=hidden_size,
                                             iterations=iterations, h_size=h_size, num_layers=num_layers,
                                             dropout=dropout)

elapsed_time, only_gpu = baseDynamicalTraining.train()
print(f'BaseDynamical - Elapsed Time: {elapsed_time}, Only GPU: {only_gpu}')
baseDynamicalPred, baseDynamicalPred2D = baseDynamicalTraining.eval(paths_0, paths_1)

########CachedDynamical##########
cachedDynamicalModel = GridModel(data_dimension)
cachedDynamicalOptimizer = torch.optim.Adam(cachedDynamicalModel.parameters(), lr=1e-3)

cachedDynamicalTraining = CachedDynamicalSystemModel("CachedDynamical", cachedDynamicalModel, run, config, dataloader,
                                                     data_dimension,
                                                     cachedDynamicalOptimizer, loss_fn, path_length=path_length,
                                                     hidden_size=hidden_size,
                                                     iterations=iterations, h_size=h_size, num_layers=num_layers,
                                                     dropout=dropout)

elapsed_time, only_gpu = cachedDynamicalTraining.train()
print(f'CachedDynamical - Elapsed Time: {elapsed_time}, Only GPU: {only_gpu}')
cachedDynamicalPred, cachedDynamicalPred2D = cachedDynamicalTraining.eval(paths_0, paths_1)

########IncreasedIterations##########
increasedIterationsModel = GridModel(data_dimension)
increasedIterationsOptimizer = torch.optim.Adam(increasedIterationsModel.parameters(), lr=1e-3)

increasedIterationsTraining = IncreasedIterationsModel("IncreasedIterations", increasedIterationsModel, run, config,
                                                       dataloader, data_dimension,
                                                       increasedIterationsOptimizer, loss_fn, path_length=path_length,
                                                       hidden_size=hidden_size,
                                                       iterations=iterations, h_size=h_size, num_layers=num_layers,
                                                       dropout=dropout)

elapsed_time, only_gpu = increasedIterationsTraining.train()
print(f'IncreasedIterations - Elapsed Time: {elapsed_time}, Only GPU: {only_gpu}')
increasedIterationsPred, increasedIterationsPred2D = increasedIterationsTraining.eval(paths_0, paths_1)

path_diff = paths_1['orig'][0] - lstmPred[0]
for i in range(63):
    path_diff = path_diff + paths_1['orig'][i+1] - lstmPred[i+1]
plt.plot(torch.norm(path_diff / 64.0, dim=1), c='g', label='LSTM_pred')

path_diff = paths_1['orig'][0] - gruPred[0]
for i in range(63):
    path_diff = path_diff + paths_1['orig'][i+1]- gruPred[i+1]
plt.plot(torch.norm(path_diff / 64.0, dim=1), c='b', label='GRU_pred')

path_diff = paths_1['orig'][0] - baseDynamicalPred[0]
for i in range(63):
    path_diff = path_diff + paths_1['orig'][i+1] - baseDynamicalPred[i+1]
plt.plot(torch.norm(path_diff / 64.0, dim=1), c='m', label='BaseDynamical_pred')

path_diff = paths_1['orig'][0] - cachedDynamicalPred[0]
for i in range(63):
    path_diff = path_diff + paths_1['orig'][i+1] - cachedDynamicalPred[i+1]
plt.plot(torch.norm(path_diff / 64.0, dim=1), c='c', label='CachedDynamical_pred')

path_diff = paths_1['orig'][0] - increasedIterationsPred[0]
for i in range(63):
    path_diff = path_diff + paths_1['orig'][i+1] - increasedIterationsPred[i+1]
plt.plot(torch.norm(path_diff / 64.0, dim=1), c='y', label='IncreasedIterations_pred')

path_diff = paths_1['orig'][0]-falconLayer0Pred[0]
for i in range(63):
    path_diff = path_diff + paths_1['orig'][i+1] - falconLayer0Pred[i+1]
plt.plot(torch.norm(path_diff / 64.0, dim=1), c='r', label='FalconLayer0_pred')

plt.legend()
plt.title("Error in prediction full dimension")

plt.savefig("error_comparison.png")
run.log({f'error_comparison': wandb.Image("error_comparison.png")})
