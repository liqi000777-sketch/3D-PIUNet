from torch import nn
import torch
"""
Copied from https://github.com/vgsatorras/egnn/blob/main/models/egnn_clean/egnn_clean.py

E(n) Equivariant Graph Neural Networks
Victor Garcia Satorras, Emiel Hogeboom, Max Welling
https://arxiv.org/abs/2102.09844

"""

class E_GCL(nn.Module):
    """
    E(n) Equivariant Convolutional Layer
    re
    """

    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, act_fn=nn.SiLU(), residual=True, attention=False, normalize=False, coords_agg='mean', tanh=False):
        super(E_GCL, self).__init__()
        input_edge = input_nf * 2
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.coords_agg = coords_agg
        self.tanh = tanh
        self.epsilon = 1e-8
        edge_coords_nf = 1

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf))

        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        coord_mlp = []
        coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
        coord_mlp.append(act_fn)
        coord_mlp.append(layer)
        if self.tanh:
            coord_mlp.append(nn.Tanh())
        self.coord_mlp = nn.Sequential(*coord_mlp)

        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())

    def edge_model(self, source, target, radial, edge_attr):
        if edge_attr is None:  # Unused.
            out = torch.cat([source, target, radial], dim=1)
        else:
            out = torch.cat([source, target, radial, edge_attr], dim=1)
        out = self.edge_mlp(out)
        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val
        return out

    def edge_model_batched(self, source, target, radial, edge_attr):
        b = source.shape[0]
        if edge_attr is None:  # Unused.
            out = torch.cat([source, target, radial[None].expand(b,-1,-1)], dim=2)
        else:
            out = torch.cat([source, target, radial[None].expand(b,-1,-1), edge_attr[None].expand(b,-1,-1)], dim=2)
        out = self.edge_mlp(out)
        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val
        return out

    def node_model_batched(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index
        agg = unsorted_segment_sum_batched(edge_attr, row, num_segments=x.size(1), batch_size=x.size(0))
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=2)
        out = self.node_mlp(agg)
        if self.residual:
            out = x + out
        return out, agg


    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = self.node_mlp(agg)
        if self.residual:
            out = x + out
        return out, agg

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        row, col = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat)
        if self.coords_agg == 'sum':
            agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
        elif self.coords_agg == 'mean':
            agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        else:
            raise Exception('Wrong coords_agg parameter' % self.coords_agg)
        coord = coord + agg
        return coord

    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum(coord_diff**2, 1).unsqueeze(1)

        if self.normalize:
            norm = torch.sqrt(radial).detach() + self.epsilon
            coord_diff = coord_diff / norm

        return radial, coord_diff

    def forward(self, h, edge_index, coord, edge_attr=None, node_attr=None):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)

        # We want to take care of batchwise opertions with the same edge model!
        if len(h.shape)==3:
            # Batchwise
            edge_feat = self.edge_model_batched(h[:,row], h[:,col], radial, edge_attr)
            h, agg = self.node_model_batched(h, edge_index, edge_feat, node_attr)
            return h, coord, edge_attr

        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)
        # We do not update coordinates, as brain sources are fix!
        #coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)

        return h, coord, edge_attr


class EGNN(nn.Module):
    def __init__(self, in_node_nf, hidden_nf, out_node_nf, in_edge_nf=0, device='cpu', act_fn=nn.SiLU(), n_layers=4, residual=True, attention=False, normalize=False, tanh=False):
        '''

        :param in_node_nf: Number of features for 'h' at the input
        :param hidden_nf: Number of hidden features
        :param out_node_nf: Number of features for 'h' at the output
        :param in_edge_nf: Number of features for the edge features
        :param device: Device (e.g. 'cpu', 'cuda:0',...)
        :param act_fn: Non-linearity
        :param n_layers: Number of layer for the EGNN
        :param residual: Use residual connections, we recommend not changing this one
        :param attention: Whether using attention or not
        :param normalize: Normalizes the coordinates messages such that:
                    instead of: x^{l+1}_i = x^{l}_i + Σ(x_i - x_j)phi_x(m_ij)
                    we get:     x^{l+1}_i = x^{l}_i + Σ(x_i - x_j)phi_x(m_ij)/||x_i - x_j||
                    We noticed it may help in the stability or generalization in some future works.
                    We didn't use it in our paper.
        :param tanh: Sets a tanh activation function at the output of phi_x(m_ij). I.e. it bounds the output of
                        phi_x(m_ij) which definitely improves in stability but it may decrease in accuracy.
                        We didn't use it in our paper.
        '''

        super(EGNN, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers

        self.embedding_in = nn.Linear(in_node_nf, self.hidden_nf)

        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, E_GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf,
                                                act_fn=act_fn, residual=residual, attention=attention,
                                                normalize=normalize, tanh=tanh))
        self.to(self.device)

    def forward(self, node_feat, coords, edges, edge_attr):
        """
        For EEG Data we have the following:
        NodeFeatures: EEG Measurements on Channel and EEG Pseudo Inverse on Nodes
        EdgeFeatures: Forward/Inverse Weight for edges between Sources and Sensors
                      Distances between Sources?
        Coordinates: 3D Coordinates of Sensors/Sources
        """
        # Transform node_feature to flatten batch
        #h = node_feat.view(b * n, f)
        # This would also work batchwise?!
        # We keep batch dimension in place as the whole batch has the same graph structure
        h = self.embedding_in(node_feat)

        for i in range(0, self.n_layers):
            h, _, _ = self._modules["gcl_%d" % i](h, edges, coords, edge_attr=edge_attr)
        h = self.embedding_out(h)
        return h


def get_edges_EEG(sources, sensors, F, Inv, dist_cutoff = 0.02):
    """
    For a fixed set of source locations and sensors, return the edges:
    1. Connecting all Sensors to all sources (with weight F[i,j] and Inv[i,j])
    2. Connecting neighboring sources with distance as edge attribute
    """
    n_sources = sources.shape[0]
    n_sensors = sensors.shape[0]

    # Edge connections between sensors and sources
    sensor_indices = torch.arange(n_sensors) + n_sources
    edges = torch.cartesian_prod(torch.arange(n_sources), sensor_indices)
    sensor_distances = torch.norm(sources[edges[:, 0]] - sensors[edges[:, 1]-n_sources], dim=1)
    edge_attr = torch.cat([
        F[edges[:,1]-n_sources,edges[:,0]].flatten(1),
        Inv[edges[:,1]-n_sources,edges[:,0]].flatten(1),
        sensor_distances.unsqueeze(1),
        torch.zeros(edges.shape[0], 1, dtype=torch.float),
        torch.ones(edges.shape[0], 1, dtype=torch.float)
    ], dim=1)
    # Pairwise distances between sources
    source_combinations = torch.combinations(torch.arange(n_sources), with_replacement=False)
    source_distances = torch.norm(sources[source_combinations[:, 0]] - sources[source_combinations[:, 1]], dim=1)

    # Filter out pairs with distance less than cutoff
    mask = source_distances < dist_cutoff
    source_combinations = source_combinations[mask]
    source_distances = source_distances[mask]

    # Edge connections between neighboring sources
    edges = torch.cat([edges, source_combinations], dim=0)
    edge_attr_within = torch.cat([
            torch.norm(F[:, source_combinations[:, 0]] - F[:, source_combinations[:, 1]], dim=0),
            torch.norm(Inv[:, source_combinations[:, 0]] - Inv[:, source_combinations[:, 1]], dim=0),
            source_distances.unsqueeze(1),
            torch.ones(source_distances.numel(), 1),
            torch.zeros(source_distances.numel(), 1)
        ], dim=1)

    edge_attr = torch.cat([
        edge_attr, edge_attr_within], dim=0)
    return edges.t(), edge_attr.float()

    edges = []
    edge_attr = []
    for i in range(n_sources):
        for j in range(n_sensors):
            edges.append([i, j + n_sources])
            edge_attr.append(torch.cat([F[j, i].flatten(), Inv[j, i].flatten(),torch.tensor([0,0,1])], dim=0))


    for i in range(n_sources):
        for j in range(i + 1, n_sources):
            dist = torch.norm(sources[i] - sources[j])
            if dist < dist_cutoff:
                edges.append([i, j])
                edge_attr.append(torch.cat([torch.norm((F[:, i]-F[:, j]), dim=0).flatten(),
                                  torch.norm(Inv[:, i]-Inv[:, j], dim=0).flatten(),torch.tensor([dist,1,0])]))
    return torch.tensor(edges).t(), torch.tensor(edge_attr)

def unsorted_segment_sum(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result

def unsorted_segment_sum_batched(data, segment_ids, num_segments,batch_size):
    result_shape = (batch_size, num_segments, data.size(2))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).unsqueeze(0).expand(batch_size,-1, data.size(2))
    result.scatter_add_(1, segment_ids, data)
    return result

def unsorted_segment_mean(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)


def get_edges(n_nodes):
    rows, cols = [], []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                rows.append(i)
                cols.append(j)

    edges = [rows, cols]
    return edges


def get_edges_batch(n_nodes, batch_size):
    edges = get_edges(n_nodes)
    edge_attr = torch.ones(len(edges[0]) * batch_size, 1)
    edges = [torch.LongTensor(edges[0]), torch.LongTensor(edges[1])]
    if batch_size == 1:
        return edges, edge_attr
    elif batch_size > 1:
        rows, cols = [], []
        for i in range(batch_size):
            rows.append(edges[0] + n_nodes * i)
            cols.append(edges[1] + n_nodes * i)
        edges = [torch.cat(rows), torch.cat(cols)]
    return edges, edge_attr


