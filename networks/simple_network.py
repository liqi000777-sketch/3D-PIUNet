import torch.nn as nn

class SimpleNetwork(nn.Module):
    # A Simple Network with Linear Layers
    def __init__(self, input_dim, output_dim, hidden_dim=64, num_layers=4, batch_norm = False):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(input_dim)
        else:
            self.batch_norm = None
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        return self.layers(x)
