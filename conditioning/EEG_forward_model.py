import torch.nn as nn
import numpy as np
import torch
import warnings
import re

from concurrent.futures import ThreadPoolExecutor
from conditioning.utils import reshape_fortran, get_mne_src_fwd_inv, save_gamma_map, total_variation_3d, champ_vec, mkfilt_eloreta_v2
from sklearn.linear_model import Lasso, LassoLars

import mne
class EEGforward(nn.Module):
    def __init__(self, mode="single", padding=False, device=None):
        super().__init__()
        self.data_dir = "data/"
        self.device = device
        self.min_snr = 1
        self.max_snr = 25
        self.TC = None
        self.mne_obj = None
        self.correlated_noise = False
        self.forward_noise = False
        self.padding = padding
        self.order = "F"
        self.n_channel = 3
        self.mode = mode
        self.snr = None
        self.morph = None
        self.load_MNE_VolumeSpace(mode)
        self.order = "C"
        return



    def set_mne_obj(self, eeg_data):
        # Possibility to set MNE objective with more data for improved covariance matrix
        self.mne_obj = get_mne_src_fwd_inv(self.mode, eeg_data=eeg_data.detach().cpu().numpy()[..., None])
    def compute_mne_inverse(self, x, method="MNEeLORETA", force_recompute=False):
        """
        Compute MNE Inverse
        x: torch.Tensor of shape (Batch, Nsens)
        method: str, MNE method (“MNE” | “dSPM” | “sLORETA” | “eLORETA”)
        """
        # Unfortunately MNE cannot do Batch Wise calculations and torch tensor
        if self.mne_obj is None or force_recompute:
            self.mne_obj = get_mne_src_fwd_inv(self.mode, eeg_data = x.detach().cpu().numpy()[..., None])

        # %% Source space
        snr =5# (self.min_snr + self.max_snr) / 2
        lambda2 = 1.0 / snr ** 2
        eeg_tmp = mne.EpochsArray(x.detach().cpu().numpy()[..., None], self.mne_obj["info"], verbose=0)
        eeg_tmp.set_eeg_reference(projection=True, verbose=0)

        if method in ["MxNE", "mneGammaMap"]:
            # TODO this is horrible slow
            results = np.zeros((x.shape[0], self.Nsources, self.n_channel))
            for i in range(len(eeg_tmp)):
                if method == "mneGammaMap":
                    dipoles = mne.inverse_sparse.gamma_map(
                        eeg_tmp[i].average(),
                        self.mne_obj["fwd"],
                        self.mne_obj["cov"],
                        0.01,
                        xyz_same_gamma=True,
                        return_residual=False,
                        return_as_dipoles=True,
                        verbose=False,
                    )
                else:
                    dipoles = mne.inverse_sparse.mixed_norm(
                        eeg_tmp[i].average(),
                        self.mne_obj["fwd"],
                        self.mne_obj["cov"],
                        loose=1,
                        maxit=2000,
                        tol=1e-4,
                        return_as_dipoles=True,
                        n_mxne_iter=2,
                        verbose=False,
                        alpha=1,
                    )
                # Dipoles contain the dipoles active, we need to transform it into the correct shape
                for dipol in dipoles:
                    results[i, np.where(np.linalg.norm(dipol.pos - self.scs_loc, axis=1) < 0.0001)[0][0]] = dipol.ori * dipol.amplitude
        elif method == "beamformer":
            if "filters" not in self.mne_obj.keys():
                self.mne_obj["filters"] = mne.beamformer.make_lcmv(
                    self.mne_obj["info"],
                    self.mne_obj["fwd"],
                    self.mne_obj["cov"], #TODO we should add some noise covariance here
                    reg=0.05,
                    pick_ori="vector",
                    rank=None,
                    weight_norm=None,
                    reduce_rank=False,
                    verbose=False,
                )
            results = [stc.data[...,0] for stc in mne.beamformer.apply_lcmv_epochs(eeg_tmp, self.mne_obj["filters"], verbose=False)]
        else:
            method = method.replace("MNE", "")
            results = [stc.data[..., 0] for stc in mne.minimum_norm.apply_inverse_epochs(eeg_tmp, self.mne_obj["inv"], lambda2, method, pick_ori="vector", verbose=0)]
        return torch.tensor(np.asarray(results), device=x.device, dtype=x.dtype)

    def load_MNE_VolumeSpace(self, mode):
        if "val" in mode:
            data_dict = np.load(f"{self.data_dir}fsaverage_volume_val.npy", allow_pickle=True)[()]
            self.instances = [1]
        elif "thingseeg" in mode:
            data_dict = np.load(f"{self.data_dir}fsaverage_volume_thingseeg.npy", allow_pickle=True)[()]
            self.instances = [0]
        elif "SUBJECT" in mode:
            subject = mode.split("SUBJECT")[-1]
            # This could be used as a pseudo Inverse for the forward model
            data_dict = np.load(f"{self.data_dir}{subject}/HeadInformationtrain-fwd.npy", allow_pickle=True)[()]
            self.instances = [0]
            self.morph = torch.Tensor(data_dict["morph"].todense()).float()
        else:
            data_dict = np.load(f"{self.data_dir}fsaverage_volume.npy", allow_pickle=True)[()]
            self.instances = [0]


        self.source_inds = data_dict["source_inds"]
        self.scs_loc = data_dict["grid_loc"]
        self.Nsens = torch.Tensor(data_dict["LeadField"]).shape[0]
        self.source_mask = torch.clamp(torch.Tensor(data_dict["source_mask"]), max=1)
        self.sens_loc = data_dict["sens_loc"]
        self.Ngridvox_xyz = (29, 29, 29)
        # The 3d locations of the voxels
        self.Ngridvox = np.prod(self.Ngridvox_xyz)
        self.img_shape = self.Ngridvox_xyz
        # Distance matrix in sparse formulation!
        self.dis_matrix = torch.Tensor(np.sum((self.scs_loc[None] - self.scs_loc[:, None]) ** 2, axis=2) ** 0.5)
        # We pad to 32x32x32 Volume
        self.pad_size = (0, 3, 0, 3, 0, 3)
        self.Nsources = len(self.source_inds)

        P = torch.Tensor(data_dict["PseudoInv"].reshape(self.Nsens, self.Nsources, -1)).float()
        L = torch.Tensor(data_dict["LeadField"].reshape(self.Nsens, self.Nsources, -1)).float()
        if self.device is not None:
            P = P.to(self.device)
            L = L.to(self.device)
            self.morph = self.morph.to(self.device) if self.morph is not None else None
            self.source_mask = self.source_mask.to(self.device)

        self.forward_matrices = {k: {"L": L,
                                     "P": P,
                                     "source_mask": self.source_mask,
                                     "Loc": torch.Tensor(self.scs_loc).float()
                                     } for k in self.instances
                                 }
        if "both" in mode or "SUBJECT" in mode:
            if "both" in mode:
                val_dict = np.load(f"{self.data_dir}fsaverage_volume_val.npy", allow_pickle=True)[()]
            else:
                val_dict = np.load(f"{self.data_dir}{subject}/HeadInformationval-fwd.npy", allow_pickle=True)[()]
            self.instances.append(1)

            self.forward_matrices[1] = {"L": torch.Tensor(val_dict["LeadField"]).float(),
                                        "P": torch.Tensor(val_dict["PseudoInv"]).float(),
                                        "source_mask": torch.clamp(torch.Tensor(val_dict["source_mask"]), max=1),
                                        "Loc": torch.Tensor(val_dict["grid_loc"]).float()
                                        }
            if self.device is not None:
                self.forward_matrices[1]["L"] = self.forward_matrices[1]["L"].to(self.device)
                self.forward_matrices[1]["P"] = self.forward_matrices[1]["P"].to(self.device)
                self.forward_matrices[1]["source_mask"] = self.forward_matrices[1]["source_mask"].to(self.device)

    def forward_specific(self, x, forward_index, **kwargs):
        """We can provide specific forward passes using a Key of the form:
        NL{Noiselevel}Cor{Correlation}ID{Index}
        for Noiselevel in [2, 5, 10, 15, 20, 40]
        Correlation in [0, 1, 2, 4, 8]
        and Index in [0,...,9]
        """
        if len(x.shape) == 5:
            x_flat = self.volume_to_vector(x).flatten(1)
        else:
            x_flat = x.flatten(1)
        if isinstance(forward_index, torch.Tensor):
            forward_index = [idx.item() for idx in forward_index]
            assert all([idx in self.forward_matrices.keys() for idx in forward_index]), f"Key {forward_index} not found"

            forward_matrix = torch.stack([self.forward_matrices[idx]["L"].flatten(1) for idx in forward_index], dim=0)
            if self.forward_noise:
                forward_matrix = self.noise_forward_matrix(forward_matrix)
            if forward_matrix.device != x.device:
                forward_matrix = forward_matrix.to(x.device)
            out_flat = torch.einsum("BD, BMD -> BM", x_flat, forward_matrix)
            return out_flat

        if forward_index not in self.forward_matrices.keys():
            warnings.warn(f"Key {forward_index} not found")
            forward_index = self.instances[0]

        forward_matrix = self.forward_matrices[forward_index]["L"].flatten(1)
        if forward_matrix.device != x.device:
            forward_matrix = forward_matrix.to(x.device)
            self.forward_matrices[forward_index]["L"] = self.forward_matrices[forward_index]["L"].to(x.device)

        if self.forward_noise:
            forward_matrix = self.noise_forward_matrix(forward_matrix)

        out_flat = torch.einsum("BD, MD -> BM", x_flat, forward_matrix)
        return out_flat

    def noise_forward_matrix(self, forward_matrix):
        # We add noise to the forward matrix with self.forward_noise percentage
        noise = torch.randn_like(forward_matrix)
        forward_std = torch.std(forward_matrix, keepdim=True)
        noise_std = forward_std * (self.forward_noise)**0.5

        return forward_matrix + noise * noise_std

    def compute_noisy_forward(self, x, forward_index=""):

        snr_multiplier = (self.max_snr - self.min_snr)
        self.snr = torch.rand((x.shape[0], 1), device=x.device) * snr_multiplier + self.min_snr

        if self.correlated_noise:
            noise = torch.randn_like(x)
            sources_std = torch.std(x, keepdim=True)
            noise_std = sources_std * (self.correlated_noise ** 0.5)
            x = x + noise * noise_std

        if forward_index in self.forward_matrices.keys():
            sensors = self.forward_specific(x, forward_index)
        else:
            raise ValueError(f"Unknown forward_index {forward_index}")
        # Compute the standard deviation of the signal
        signal_std = torch.std(sensors, dim=1, keepdim=True)
        # Compute the standard deviation of the noise
        noise_std = signal_std / (10 ** (self.snr/10)) ** 0.5
        sensors = sensors + torch.randn_like(sensors) * noise_std
        return sensors

    def set_noise_pattern(self, forward_noise=False, correlated_noise=False):
        self.forward_noise = forward_noise
        self.correlated_noise = correlated_noise

    def set_snr(self, min_snr, max_snr):
        self.min_snr = min_snr
        self.max_snr = max_snr
    def remove_padding(self, x):
        return x[:, :, : self.Ngridvox_xyz[0], : self.Ngridvox_xyz[1], : self.Ngridvox_xyz[2]]

    def add_padding(self, x):
        return nn.functional.pad(x, self.pad_size)

    def pseudo_inv_specific(self, x, forward_index, return_volume=False, mode="loaded"):
        """
        Assumes input of Shape (Batch, M)
        Returns Pseudo Inverse of Shape (Batch, D, C)
        """
        if mode == "loaded":
            if isinstance(forward_index, torch.Tensor):
                forward_index = [idx.item() for idx in forward_index]
                assert all([idx in self.forward_matrices.keys() for idx in forward_index]), f"Key not found!, {forward_index}"
                pseudo_inv = torch.stack([self.forward_matrices[idx]["P"] for idx in forward_index], dim=0)
                if pseudo_inv.device != x.device:
                    pseudo_inv = pseudo_inv.to(x.device)
                out_flat = torch.einsum("BM, BMDC -> BDC", x, pseudo_inv)
            else:
                if forward_index not in self.forward_matrices.keys():
                    warnings.warn(f"Key {forward_index} not found")
                    forward_index = self.instances[0]
                    #return self.pseudo_inv(x)
                pseudo_inv = self.forward_matrices[forward_index]["P"]
                if pseudo_inv.device != x.device:
                    pseudo_inv = pseudo_inv.to(x.device)
                    self.forward_matrices[forward_index]["P"] = self.forward_matrices[forward_index]["P"].to(x.device)

                out_flat = torch.einsum("BM, MDC -> BDC", x, pseudo_inv)
        else:
            assert not isinstance(forward_index, torch.Tensor), "Pseudo Inverse only implemented for homogenous batch-forward"
            if mode == "monroe_penrose":
                out_flat = self.monroe_penrose_inverse(x, forward_index)
            elif mode == "gamma":
                out_flat = self.gamma_inverse(x, forward_index)
            elif mode == "lasso":
                out_flat = self.pytorch_lasso(x, forward_index)
            elif mode in ["MNEeLORETA","MNE","dSPM","sLORETA", "MxNE", "beamformer"]:
                out_flat = self.compute_mne_inverse(x, method=mode)
            else:
                raise ValueError(f"Unknown mode for pseudo inverse {mode}")
        if return_volume:
            return self.vector_to_volume(out_flat)
        return out_flat



    def monroe_penrose_inverse(self, x, forward_index):
        """
        Takes a tensor of shape (Batch, M) and returns the pseudoinverse prediction using Moore Penrose algorithm
        """

        if "MP_inv" not in self.forward_matrices[forward_index].keys():
            self.forward_matrices[forward_index]["MP_inv"] = torch.linalg.pinv(self.forward_matrices[forward_index]["L"].flatten(1))
        out = self.forward_matrices[forward_index]["MP_inv"].to(x.device) @ x.transpose(1, 0)
        return out.reshape(-1, self.n_channel, x.shape[0]).permute(2, 0, 1)

    def champagne(self, x, forward_index, alpha=0.05):
        sens = x.cpu().detach().numpy()

        if self.mne_obj is None:
            self.mne_obj = get_mne_src_fwd_inv(self.mode, eeg_data=sens[..., None])

        noise_var = self.mne_obj["cov"].data
        out = []
        L = self.forward_matrices[forward_index]["L"].flatten(1).numpy()
        # Champagne expects input of shape: N_sensors, N_time

        for s in sens: # We do for every timestep individually
            out.append(champ_vec(s[..., None], L, noise_var * alpha, 400, self.n_channel).reshape(self.Nsources,self.n_channel))

        out = np.asarray(out)
        return torch.tensor(out, device=x.device, dtype=torch.float32)

    def gamma_inverse(self, x, forward_index, alpha=.2):
        """
        Takes a tensor of shape (Batch, M) and returns the pseudoinverse prediction using gamma map algorithm
        (Probably not working correctly atm!)
        """

        sens = x.cpu().detach().numpy()
        if self.mne_obj is None:
            self.mne_obj = get_mne_src_fwd_inv(self.mode, eeg_data=sens[..., None])

        if False:
            # Old compute of variance
            snr = np.random.rand() * (self.max_snr - self.min_snr) + self.min_snr

            #.transpose(1, 0)
            signal_std = np.std(sens, axis=0)
            noise_var = signal_std ** 2 / (10 ** (snr))
            noise_var = np.diag(noise_var)
        else:
            noise_var = self.mne_obj["cov"].data

        #Unfortunately gammamap does not support batched operation, so we need to loop
        L = self.forward_matrices[forward_index]["L"].flatten(1).numpy()
        if False:
            with ThreadPoolExecutor(max_workers=10) as executor:
                out = list(
                    executor.map(lambda s: save_gamma_map(s[...,None], L, noise_var, self.n_channel, self.Nsources),sens)
                )
        else:
            out = []
            for s in sens:
                out.append(save_gamma_map(s[..., None], L, noise_var, self.n_channel, self.Nsources, alpha=alpha))
        out = np.asarray(out)[..., 0]
        #out = gamma_map(L=L, y=sens, n_orient=self.n_channel, alpha=0.6, cov=noise_var)

        #return self.vector_to_volume(torch.tensor(out.transpose(2,0,1), device=x.device, dtype=torch.float32))
        return torch.tensor(out, device=x.device, dtype=torch.float32)

    def pytorch_lasso(self, sensor, forward_index, lambda_lasso=150, max_iter=4000, united_pos=False,
                      start_pseudo=True):
        pred_source = self.pseudo_inv_specific(sensor, forward_index).detach()

        with torch.enable_grad():
            if start_pseudo:
                pred_source = torch.nn.Parameter(pred_source, requires_grad=True)
            else:
                pred_source = torch.nn.Parameter(torch.zeros_like(pred_source, requires_grad=True), requires_grad=True)

            optimizer = torch.optim.Adam([pred_source], lr=0.01)
            best_loss = torch.inf
            best_loss_delay = 0
            for i in range(max_iter):  # Adjust the number of iterations as needed
                optimizer.zero_grad()
                scaled_out = self.forward_specific(pred_source, forward_index)
                mse_loss = torch.nn.functional.mse_loss(scaled_out, sensor)
                if united_pos:
                    # L1 Regularization for Position Norm
                    l1_penalty = lambda_lasso * torch.norm(pred_source.norm(dim=2), dim=1,
                                                           p=1).mean()
                else:
                    # L1 regularization
                    l1_penalty = lambda_lasso * torch.norm(pred_source.flatten(1), dim=1, p=1).mean()

                loss = mse_loss + l1_penalty
                loss.backward()
                optimizer.step()
                if loss > best_loss:
                    best_loss_delay += 1
                    if best_loss_delay >= 200:
                        break
                else:
                    best_loss_delay = 0
                    best_loss = loss.detach()
        # print(f"It took {i} iterations to converge")
        return pred_source.detach()

    def pytorch_tv(self, sensor, forward_index, lambda_reg=150, max_iter=2000):
        pred_source = self.pseudo_inv_specific(sensor, forward_index, return_volume=True).detach()

        with torch.enable_grad():

            #pred_source = torch.nn.Parameter(torch.zeros_like(pred_source, requires_grad=True), requires_grad=True)
            pred_source = torch.nn.Parameter(pred_source, requires_grad=True)

            optimizer = torch.optim.Adam([pred_source], lr=0.01)
            best_loss = torch.inf
            best_loss_delay = 0
            for i in range(max_iter):  # Adjust the number of iterations as needed
                optimizer.zero_grad()
                scaled_out = self.forward_specific(pred_source, forward_index)
                mse_loss = torch.nn.functional.mse_loss(scaled_out, sensor)
                tv_reg = lambda_reg * total_variation_3d(pred_source.norm(dim=1))

                loss = mse_loss + tv_reg
                loss.backward()
                optimizer.step()
                if loss > best_loss:
                    best_loss_delay += 1
                    if best_loss_delay >= 200:
                        break
                else:
                    best_loss_delay = 0
                    best_loss = loss.detach()
        #print(f"It took {i} iterations to converge")
        return pred_source.detach()

    def lasso_inverse(self, x, forward_index, mode="Lasso",lambda_lasso=5):
        """
        Takes a tensor of shape (Batch, M) and returns the pseudoinverse prediction using lasso algorithm

        """
        if mode == "lassoLars":
            clf = LassoLars(alpha=lambda_lasso, max_iter=20000)
        else:
            clf = Lasso(alpha=lambda_lasso, max_iter=20000)
        sens = x.cpu().detach().numpy().transpose(1, 0)
        clf.fit(self.forward_matrices[forward_index]["L"].flatten(1).numpy(), sens)
        out = clf.coef_
        return torch.tensor(out, device=x.device, dtype=torch.float32).reshape(x.shape[0], -1, self.n_channel)

    def vector_to_volume(self, x):
        """
        Converts Vector to Volume
        """
        if self.TC is not None:
            out_volume = self.TC.spread_to_grid(x)
        else:
            x_channel = x.shape[2]
            out_volume = torch.zeros((x.shape[0], self.Ngridvox, x_channel), device=x.device, dtype=x.dtype)
            out_volume[:, self.source_inds] = x.reshape(x.shape[0], -1, x_channel)
            if self.order =="F":
                out_volume = reshape_fortran(
                    out_volume, (-1, self.Ngridvox_xyz[0], self.Ngridvox_xyz[1], self.Ngridvox_xyz[2], x_channel)
                )
            else:
                out_volume = torch.reshape(out_volume,(-1, self.Ngridvox_xyz[0], self.Ngridvox_xyz[1], self.Ngridvox_xyz[2], x_channel))
        # Move channel from last to second dimension
        out_volume = out_volume.permute(0, -1, 1, 2, 3)
        if self.padding:
            out_volume = nn.functional.pad(out_volume, self.pad_size)
        return out_volume

    def matrix_to_grid(self, L):
        # vector_to_volume that Works for Numpy Forward Matcies without Batch
        Nsens = L.shape[0]
        grid = np.zeros((Nsens, self.Ngridvox, self.n_channel))
        grid[:, self.source_inds] = L
        grid = np.reshape(grid, (Nsens, self.Ngridvox_xyz[0], self.Ngridvox_xyz[1], self.Ngridvox_xyz[2], self.n_channel),
                          order=self.order)
        return torch.Tensor(np.transpose(grid, (0, 4, 1, 2, 3)))

    def volume_to_vector(self, x):
        """
        Converts volume to vector
        Returns vector of shape: Batchsize, N_Sources, N_Channel
        """
        if self.padding:
            x = x[:, :, : self.Ngridvox_xyz[0], : self.Ngridvox_xyz[1], : self.Ngridvox_xyz[2]]

        if self.TC is not None:
            return self.TC.grid_to_spread(x.flatten(2)).permute(0,2,1)

        n_channel = x.shape[1]
        # Move channel from second to last dimension
        x = x.permute(0, 2, 3, 4, 1)


        if self.order =="F":
            x = reshape_fortran(x, (-1, self.Ngridvox_xyz[0] * self.Ngridvox_xyz[1] * self.Ngridvox_xyz[2], n_channel))
        else:
            x = torch.reshape(x,  (-1, self.Ngridvox_xyz[0] * self.Ngridvox_xyz[1] * self.Ngridvox_xyz[2], n_channel))
        return x[:, self.source_inds]

    def apply_morph(self, x):
        if self.morph is None:
            return x.reshape(x.shape[0], self.Nsources, -1)
        if self.morph.device != x.device:
            self.morph = self.morph.to(x.device)

        x = torch.einsum("bsc,ns->bnc",
                           x.reshape(x.shape[0], self.morph.shape[-1], -1),
                           self.morph)
        x = x.reshape(x.shape[0], self.Nsources, -1)
        return x

    def get_resoultion_kernel(self, forward_index="", large=False):
        # TODO Implement for forward_indices
        if large:  # BIG forward Matrix
            forward_matrix = torch.einsum(
                "MABCD , MEFGH -> ABCDEFGH", self.Pgrid, self.Lgrid
            )
        else:  # Small forward Matix with reshape necessary
            forward_matrix = torch.einsum(
                "AB, AC -> BC", self.P.flatten(1), self.L.flatten(1)
            )

        return forward_matrix

    def get_volume_psuedo_inv(self):
        #TODO Test that small matrix and large are the same
        pseudo_L = self.get_resoultion_kernel()
        return self.vector_to_volume(self.vector_to_volume(pseudo_L).flatten(1).T).T.view(19,23,20,3,3, 20, 23, 19)


    def transpose(self, x):
        raise NotImplementedError("Currently Transpose not implemented for EEG forward")

    def ortho_project(self, data, **kwargs):
        # calculate (I - A^T * A)X
        return data - self.transpose(self.forward_specific(data, **kwargs), **kwargs)

    def project(self, data, measurement, **kwargs):
        # calculate (I - A^T * A)Y - AX
        return self.ortho_project(measurement, **kwargs) - self.forward_specific(data, **kwargs)

    def get_sigma_displacement(self, forward_index=""):

        if isinstance(forward_index, torch.Tensor):
            assert all([idx in self.forward_matrices.keys() for idx in forward_index]), "Key not found!"
            raise NotImplementedError("Need to verify and adapt other stuff depending on it")
            return torch.stack([self.forward_matrices[idx]["sigmaDisplace"] for idx in forward_index], dim=0)

        if forward_index not in self.forward_matrices.keys():
            warnings.warn(f"Key {forward_index} not found in get_sigma_displacement")
            return self.displace_sigma_matrix
        else:
            return self.forward_matrices[forward_index]["sigmaDisplace"]

    def get_LGrid(self, forward_index=""):
        """
        We return the Forward Grid Matrix of the correspond forward_index
        """
        # LGrid Sensors x 3 x WxHxD
        if forward_index not in self.forward_matrices.keys():
            warnings.warn(f"Key {forward_index} not found in get_LGrid")
            LGrid = self.Lgrid
        else:
            LGrid = self.vector_to_volume(self.forward_matrices[forward_index]["L"])
        return LGrid


    def get_source_mask(self, forward_index):

        try:
            return self.forward_matrices[forward_index]["source_mask"][None, None]
        except KeyError:
            return self.source_mask[None, None]


    def select_sensors_subset(self, subset):
        """
        Selects a subset of sensors, leaving the number of sources unchanged
        """
        #self.Nsens = len(subset)
        #self.sens_loc = self.sens_loc[subset]
        # Instead of selecting a subset, we just set the corresponding entries to zero
        for k in self.forward_matrices.keys():
            # We try to renormalize
            #self.forward_matrices[k]["L"] *= self.forward_matrices[k]["L"].shape[0]/len(subset)
            #self.forward_matrices[k]["P"] *= self.forward_matrices[k]["L"].shape[0]/len(subset)
            self.forward_matrices[k]["L"][subset] = 0
            #self.forward_matrices[k]["P"][subset] = 0
            # We recompute the pseudo inverse based on the new forward matrix
            # First, we get the indicies for values
            non_zero = torch.where(self.forward_matrices[k]["L"].flatten(1).sum(1) != 0)[0]
            P = torch.tensor(mkfilt_eloreta_v2(
                self.forward_matrices[k]["L"][non_zero])).float().reshape(
                -1, self.Nsources, 3)
            self.forward_matrices[k]["P"][:] = 0
            self.forward_matrices[k]["P"][non_zero] = P


