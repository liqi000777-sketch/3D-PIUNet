from torch import nn
# Code adapted from https://github.com/bfinl/DeepSIF/blob/main/network.py


class LinearResBlock(nn.Module):
    def __init__(self, in_dim, out_dim, activation):
        super(LinearResBlock, self).__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.fc2 = nn.Linear(out_dim, out_dim)
        if in_dim != out_dim:

            self.res_fc = nn.Linear(in_dim, out_dim)
        else:
            self.res_fc = nn.Identity()
        self.activation = nn.__dict__[activation]()

    def forward(self, x):
        return self.activation(self.fc2(self.activation(self.fc1(x))) + self.res_fc(x))


class MLPSpatialFilter(nn.Module):

    def __init__(self, num_sensor, num_hidden, activation):
        super(MLPSpatialFilter, self).__init__()
        self.res_block_1 = LinearResBlock(num_sensor, num_sensor, activation)
        self.res_block_2 = LinearResBlock(num_sensor, num_hidden, activation)
        self.value = nn.Linear(num_hidden, num_hidden)
        self.activation = nn.__dict__[activation]()

    def forward(self, x):
        x = self.res_block_1(x)
        x = self.res_block_2(x)
        return self.activation(self.value(x))


class TemporalFilter(nn.Module):

    def __init__(self, input_size, num_source, num_layer, activation):
        super(TemporalFilter, self).__init__()
        self.rnns = nn.ModuleList()

        self.rnns.append(nn.LSTM(input_size, num_source, batch_first=True, num_layers=num_layer))
        self.num_layer = num_layer
        self.input_size = input_size
        self.activation = nn.__dict__[activation]()

    def forward(self, x):
        # c0/h0 : num_layer, T, num_out
        for l in self.rnns:
            l.flatten_parameters()
            x, _ = l(x)

        return x


class TemporalInverseNet(nn.Module):

    def __init__(self, num_sensor=64, num_source=994, hidden_dim=500, activation='ELU'):
        super(TemporalInverseNet, self).__init__()
        # Spatial filtering
        self.batch_norm = nn.BatchNorm1d(num_sensor)
        self.spatial = MLPSpatialFilter(num_sensor, hidden_dim, activation)

        # We have no temporal model so we use a single linear layer instead:
        self.temporal = LinearResBlock(hidden_dim, hidden_dim, activation)
        self.out_block = nn.Linear(hidden_dim, num_source)

    def forward(self, x):
        x = self.batch_norm(x)
        x = self.spatial(x)
        x = self.temporal(x)
        return self.out_block(x)


class DeepSIFNet(nn.Module):

    def __init__(self, num_sensor=64, num_source=994, hidden_dim=500, rnn_layer=3,
                 spatial_model=MLPSpatialFilter, temporal_model=TemporalFilter,
                 spatial_activation='ReLU', temporal_activation='ReLU', time_steps=200):
        super(DeepSIFNet, self).__init__()
        self.attribute_list = [num_sensor, num_source, rnn_layer,
                               spatial_model, temporal_model,
                               spatial_activation, temporal_activation, hidden_dim, time_steps]
        self.time_steps = time_steps
        # Spatial filtering
        self.spatial = spatial_model(num_sensor, hidden_dim, spatial_activation)
        # Temporal filtering
        self.temporal = temporal_model(hidden_dim, hidden_dim, rnn_layer, temporal_activation)

        self.out_block = nn.Linear(hidden_dim, num_source)
    def forward(self, x, time_steps=None):
        if time_steps is None:
            time_steps = self.time_steps
        x = self.spatial(x)
        B, C = x.shape
        x = self.temporal(x.reshape(-1, time_steps, x.shape[-1])).reshape(B, -1)
        return self.out_block(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)