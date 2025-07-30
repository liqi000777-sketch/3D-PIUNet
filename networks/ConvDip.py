from mne.viz.topomap import (_setup_interp, _make_head_outlines, _check_sphere,
    _check_extrapolate)
from mne.channels.layout import _find_topomap_coords
import numpy as np
import torch
import torch.nn as nn

"""
MIT License

Copyright (c) 2021 Lukas Hecker

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

class ConvDipModel(nn.Module):
    def __init__(self, out_dim, info):
        super(ConvDipModel, self).__init__()

        self.interp_channel_shape = (12, 12)
        self.info = info
        elec_pos = _find_topomap_coords(self.info, self.info.ch_names)
        self.interpolator = self.make_interpolator(elec_pos, res=self.interp_channel_shape[0])

        # Convolutional layer with 8 filters of size 3x3
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(8)

        # Fully connected layers
        self.fc1 = nn.Linear(12*12 * 8, 512)
        self.batch_norm2 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, out_dim)

        # Activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        batch_size = x.shape[0]
        # Interpolate the input to a 2D grid
        x = self.get_grid(x)
        # Convolution + BatchNorm + ReLU
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)

        # Flatten
        x = x.view(batch_size, -1)

        # Fully connected layers + BatchNorm + ReLU
        x = self.fc1(x)
        x = self.batch_norm2(x)
        x = self.relu(x)

        # Output layer
        x = self.fc2(x)
        return x


    @staticmethod
    def make_interpolator(elec_pos, res=9, ch_type='eeg', image_interp="linear"):
        extrapolate = _check_extrapolate('auto', ch_type)
        sphere = _check_sphere(None)
        outlines = 'head'
        outlines = _make_head_outlines(sphere, elec_pos, outlines, (0., 0.))
        border = 'mean'
        extent, Xi, Yi, interpolator = _setup_interp(
            elec_pos, res, image_interp, extrapolate, outlines, border)
        interpolator.set_locations(Xi, Yi)

        return interpolator

    def get_grid(self, x):
        conv_x = []
        # save device and dtype of x
        device = x.device
        dtype = x.dtype
        # if x is a tensor, convert it to numpy
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        for i, sample in enumerate(x):
            time_slice_interp = self.interpolator.set_values(sample)()
            conv_x.append(time_slice_interp[np.newaxis,:])
        conv_x = np.stack(conv_x, axis=0)
        # replace nans with zeros
        conv_x[np.isnan(conv_x)] = 0
        return torch.tensor(conv_x).to(device, dtype=dtype)
