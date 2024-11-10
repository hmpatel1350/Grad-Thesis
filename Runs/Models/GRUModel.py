import torch
class GRUModel(torch.nn.Module):
    def __init__(self, data_dimension, hidden_size=64, num_layers=4, dropout=0.5):
        super().__init__()
        self.gru = torch.nn.GRU(data_dimension, hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.linear = torch.nn.Linear(hidden_size, data_dimension)

    def forward(self, x):
        x, _ = self.gru(x)
        x = self.linear(x)
        return x