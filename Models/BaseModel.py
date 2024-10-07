import torch
class BaseGridModel(torch.nn.Module):
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


class BatchGridModel(torch.nn.Module):
    def __init__(self, data_dimension, h_size=64, hidden_size=128):
        super().__init__()
        self.data_dimension = data_dimension
        self.h_size = h_size
        self.hidden_size = hidden_size
        self.Fin = torch.nn.Linear(in_features=self.data_dimension, out_features=self.hidden_size)

        self.Hin = torch.nn.Linear(in_features=2 * self.h_size, out_features=self.hidden_size)

        self.Out = torch.nn.Linear(in_features=self.hidden_size, out_features=self.data_dimension + self.h_size)

    def forward(self, P0, h_i, h_j):
        z = torch.cat([h_i, h_j], dim=-1)
        out1 = self.Fin(P0) + self.Hin(z)
        gelu = torch.nn.GELU()
        out2 = gelu(out1)
        out = self.Out(out2)
        return out[:, :self.data_dimension], out[:, self.data_dimension:]