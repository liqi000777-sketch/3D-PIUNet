from typing import Any

import torch
import torch.nn as nn
import numpy as np



class SimpleTransformer(nn.Module):
    """
    A Transformer that encodes the measurments and decodes in Source space
    """
    def __init__(self, channels=36, num_heads=4, num_layers=4, meas_channels=1, source_channels=3, out_channels=3, source_norm=True, norm_first=True, dec_dim=None):
        super(SimpleTransformer, self).__init__()
        self.meas_pos_emb = PositionalEncoding3D(channels)
        self.source_pos_emb = PositionalEncoding3D(channels)
        if dec_dim is None:
            dec_dim = num_layers

        self.transformer = nn.Transformer(channels, num_heads, num_encoder_layers=num_layers, num_decoder_layers=dec_dim,
                                          batch_first=True, dropout=0.1, dim_feedforward=channels*4, norm_first=norm_first)
        self.meas_in_layer = nn.Linear(meas_channels, channels)
        self.source_in_layer = nn.Linear(source_channels, channels)
        self.out_layer = nn.Linear(channels, out_channels)
        self.source_emb = None
        self.meas_emb = None
        self.meas_norm = torch.nn.BatchNorm1d(meas_channels)
        if source_norm:
            self.source_norm = torch.nn.BatchNorm1d(source_channels)
        else:
            self.source_norm = lambda x: x
        self.out_channels = out_channels
        self.source_channels = source_channels
    def set_source_embedding(self, source_loc):
        self.source_emb = self.source_pos_emb(source_loc)

    def set_meas_embedding(self, meas_loc):
        self.meas_emb = self.meas_pos_emb(meas_loc)

    def forward(self, meas, pseudo_inv, residual=False):
        if self.meas_emb is None:
            raise RuntimeError("No measurement embedding set!")
        if self.source_emb is None:
            raise RuntimeError("No source embedding set!")
        if self.meas_emb.device != meas.device:
            self.meas_emb = self.meas_emb.to(meas.device)
            self.source_emb = self.source_emb.to(meas.device)
        # The norm requires the channels to be the 2nd dimension
        x = self.meas_norm(meas.permute(0,2,1)).permute(0,2,1)
        x = self.meas_in_layer(x) + self.meas_emb[None,:]
        y = self.source_norm(pseudo_inv.permute(0,2,1)).permute(0,2,1)
        y = self.source_in_layer(y) + self.source_emb[None,:]

        if residual:
            # TODO when out_channel != source_channel, we should only add to the first source_channels
            x = self.out_layer(self.transformer(x, y))
            return nn.functional.pad(pseudo_inv,(0, 0, 0, self.out_channels-self.source_channels)) + x

        else:
            return self.out_layer(self.transformer(x,y))

    def get_hidden_rep(self, meas):
        if self.meas_emb is None:
            raise RuntimeError("No measurement embedding set!")
        if self.meas_emb.device != meas.device:
            self.meas_emb = self.meas_emb.to(meas.device)
        x = self.meas_norm(meas.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.meas_in_layer(x) + self.meas_emb[None, :]
        return self.transformer.encoder(x)

class MeasurementEncoderTransformer(nn.Module):
    def __init__(self, channels=288, num_heads=8, num_layers=3, meas_channels=1):
        super(MeasurementEncoderTransformer, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=channels, nhead=num_heads, dim_feedforward=channels*4, batch_first=True, norm_first=True)
        self.meas_pos_emb = PositionalEncoding3D(channels)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.meas_in_layer = nn.Linear(meas_channels, channels)
        self.meas_emb = None
        self.meas_norm = torch.nn.BatchNorm1d(meas_channels)

    def set_meas_embedding(self, meas_loc):
        self.meas_emb = self.meas_pos_emb(meas_loc)

    def forward(self, meas):
        if self.meas_emb is None:
            raise RuntimeError("No measurement embedding set!")
        if self.meas_emb.device != meas.device:
            self.meas_emb = self.meas_emb.to(meas.device)
        x = self.meas_norm(meas.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.meas_in_layer(x) + self.meas_emb[None, :]
        return self.transformer_encoder(x)

class MeasureUNet(nn.Module):
    def __init__(self, unet, transformer):
        super(MeasureUNet, self).__init__()
        self.unet = unet
        self.transformer = transformer

    def forward(self, meas, pseudo_inv, timesteps, y=None, return_hidden=False):
        # Encode the measurements and change their Dimensions to (Batch, Channels, N_Sensors)
        encoded_meas = self.transformer(meas).permute(0, 2, 1)
        return self.unet(pseudo_inv, timesteps, y, return_hidden=return_hidden, encoder_out=encoded_meas)

class EEGTransformer(nn.Module):
    """
    A Transformer that encodes the measurments and decodes in Source space
    We set the Attention Matries according the forward model
    -> the resolution Kernel is interpreted as  Self-Attention of the Sources
    -> the forward Model can be used as Cross-Attention for the Measurements
    """

    def __init__(self, channels=36, num_heads=4, num_layers=4, meas_channels=1, source_channels=3, out_channels=3):
        super(EEGTransformer, self).__init__()
        self.meas_pos_emb = PositionalEncoding3D(channels)
        self.source_pos_emb = PositionalEncoding3D(channels)


        self.meas_in_layer = nn.Linear(meas_channels, channels)
        self.source_in_layer = nn.Linear(source_channels, channels)
        self.out_layer = nn.Linear(channels, out_channels)
        self.source_emb = None
        self.meas_emb = None
        self.meas_norm = torch.nn.BatchNorm1d(meas_channels)
        self.out_channels = out_channels
        self.source_channels = source_channels


    def set_resolution_kernel(self, K):
        """
        Setting the Resolution Kernel K = A^-1 A  (A is the forward model)
        K has the shape (d,N,N,d) where N is the number of sources, d the number of source_drections
        """
        self.resolution_kernel = K

    def set_forward_model(self, A, Ainv):
        """
        Setting the forward model A and its pseudo inverse Ainv
        """
        self.A = A
        self.Ainv = Ainv

    def set_source_embedding(self, source_loc):
        self.source_emb = self.source_pos_emb(source_loc)

    def set_meas_embedding(self, meas_loc):
        self.meas_emb = self.meas_pos_emb(meas_loc)

    def forward(self, meas, pseudo_inv, residual=False):
        if self.meas_emb is None:
            raise RuntimeError("No measurement embedding set!")
        if self.source_emb is None:
            raise RuntimeError("No source embedding set!")
        if self.meas_emb.device != meas.device:
            self.meas_emb = self.meas_emb.to(meas.device)
            self.source_emb = self.source_emb.to(meas.device)
        x = self.meas_norm(meas.permute(0,2,1)).permute(0,2,1)
        x = self.meas_in_layer(x) + self.meas_emb[None,:]
        y = self.source_in_layer(pseudo_inv) + self.source_emb[None,:]

        if residual:
            # TODO when out_channel != source_channel, we should only add to the first source_channels
            x = self.out_layer(self.transformer(x, y))
            return nn.functional.pad(pseudo_inv,(0, 0, 0, self.out_channels-self.source_channels)) + x

        else:
            return self.out_layer(self.transformer(x,y))




def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)


class PositionalEncoding3D(nn.Module):
    def __init__(self, channels, voxel_distance=0.007):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding3D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 6) * 2)
        if channels % 2:
            channels += 1
        self.channels = channels
        dist_multiplier = 1. / voxel_distance
        inv_freq = dist_multiplier / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.register_buffer("cached_penc", None, persistent=False)

    def forward(self, positions):
        """
        For our usecase, we have the same positions for every element in the batch
        :param positions: (Nodes, 3) The 3D Coordinates of each position
        :return: Positional Encoding Matrix of size (Nodes, ch)
        """
        if positions.shape[1] != 3:
            raise RuntimeError("The input tensor has to be 3d Coordinates!")

        if self.cached_penc is not None and self.cached_penc.shape[0] == positions.shape[0]:
            # This is only valid when all the sources/Sensors stay at the same place!
            return self.cached_penc
        self.inv_freq = self.inv_freq.to(positions.device)
        self.cached_penc = None

        pos_x = positions[:,0]
        pos_y = positions[:,1]
        pos_z = positions[:,2]
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        sin_inp_z = torch.einsum("i,j->ij", pos_z, self.inv_freq)
        print(sin_inp_x.shape, sin_inp_y.shape, sin_inp_z.shape)
        emb_x = get_emb(sin_inp_x)
        emb_y = get_emb(sin_inp_y)
        emb_z = get_emb(sin_inp_z)
        print(emb_x.shape,emb_y.shape,emb_z.shape)
        emb = torch.zeros(
            (positions.shape[0], self.channels * 3),
            device=positions.device,
            dtype=positions.dtype,
        )
        emb[:, :self.channels] = emb_x
        emb[:, self.channels : 2 * self.channels] = emb_y
        emb[:, 2 * self.channels :] = emb_z

        self.cached_penc = emb
        return self.cached_penc


class PositionalEncoding1D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding1D, self).__init__()
        self.org_channels = channels
        if channels % 2:
            channels += 1
        self.channels = channels

        inv_freq = 1 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.register_buffer("cached_penc", None, persistent=False)

    def forward(self, positions):
        """
        For our usecase, we have the same positions for every element in the batch
        :param positions: (Nodes) The 1D Coordinates of each position
        :return: Positional Encoding Matrix of size (Nodes, ch)
        """
        if len(positions.shape) != 1:
            raise RuntimeError("The input tensor has to be 1d Coordinate!")

        if self.cached_penc is not None and self.cached_penc.shape[0] == positions.shape[0]:
            # This is only valid when the time stays the same
            return self.cached_penc
        self.inv_freq = self.inv_freq.to(positions.device)
        self.cached_penc = None

        pos_x = positions
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        emb = get_emb(sin_inp_x)

        self.cached_penc = emb
        return self.cached_penc


