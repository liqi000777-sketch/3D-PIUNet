import time
from argparse import ArgumentParser

import torch
import pytorch_lightning as pl
import torch.nn as nn
import os
from networks.unet import UNetModel
from networks.nn import init_weights
from conditioning.EEG_forward_model import EEGforward
from conditioning.utils import get_info
import numpy as np
import warnings
from torch.optim import AdamW, lr_scheduler
from distributions.pointdistribution import PointDistribution
from distributions.MixtureOfExperts import MixtureOfExperts, move_expert_channel
from pytorch_lightning import loggers as pl_loggers
from utils.metrics import compute_metrics
from utils.loss_utils import (WeightedHausdorffDistance, activation_weighted_mse, activation_weighted_mae,
                              activation_focal_loss, PseudoHuberLoss)
import pandas as pd
from tqdm import tqdm
from networks.base_model import BaseEEGModel
from networks.simple_network import SimpleNetwork
from networks.egnn import EGNN, get_edges_EEG
from networks.ConvDip import ConvDipModel
from networks.transformer import SimpleTransformer, MeasurementEncoderTransformer, MeasureUNet
from networks.deepsif import TemporalInverseNet, DeepSIFNet
from networks.classical_approach import ClassicalApproach

from utils.argparse_utils import add_lightning_args
import re
from utils.data_utils import save_compressed_pickle

class SupervisedEEGModel(BaseEEGModel):
    def __init__(
        self,
        args,
    ):
        super().__init__()
        self.save_hyperparameters(args)
        self.old_net = False
        self.log_dir = f"lightning_logs/{self.hparams.experiment_name}"
        try:
            self.logger.experiment.config.update(self.hparams)
        except Exception as e:
            print(f"Could not log config {e}")
        # Set number of channels depending on output probability


        if "normal" in self.hparams.probabilistic_out:
            channel_multiplier = 2
        elif "MOE" in self.hparams.probabilistic_out:
            if "#" in self.hparams.probabilistic_out:
                self.n_experts = int(self.hparams.probabilistic_out.split("#")[1])
            else:
                self.n_experts = 2
            channel_multiplier = 3 * self.n_experts
        else:
            channel_multiplier = 1


        if "MNEVol" in self.hparams.data_set:
            in_channels = 3
            out_channel = 3 * channel_multiplier
            self.img_size = 32
            self.forward_model = EEGforward("MNEVol", device=self.device)
            self.scaling_factor = torch.zeros((10))
            dims = 3
        else:
            raise NotImplementedError("Unknown Dataset")

        # First Hacky Way to deal with time!
        if "TimeFreq" in self.hparams.data_set:
            self.time_steps = int(re.search(r'TimeFreq([0-9]+)', self.hparams.data_set).group(1))
        else:
            self.time_steps = 1

        # Set Noise of the forward model
        if "min_snr" in self.hparams:
            self.forward_model.set_snr(self.hparams.min_snr, self.hparams.max_snr)
        if "forward_noise" in self.hparams and "correlated_noise" in self.hparams:
            self.forward_model.set_noise_pattern(self.hparams.forward_noise, self.hparams.correlated_noise)
        if "pseudo_inv_mode" in self.hparams:
            self.pseudo_inv_mode = self.hparams.pseudo_inv_mode
        else:
            self.pseudo_inv_mode = "loaded"


        self.cond_dim = None

        self.in_channels = in_channels
        channel_multiplier = (1,2,4,4,4,4,4)[:self.hparams.depth]
        # TODO Make the choice of Network architecture more simple
        if self.hparams.network_architecture in ["linear", "linearBN"]:
            self.eeg_net = SimpleNetwork(self.forward_model.Nsens, self.forward_model.Nsources*out_channel,
                                         self.hparams.network_size, num_layers=self.hparams.depth+1, batch_norm="BN" in self.hparams.network_architecture)
            self.scaling_factor = 1.0
        elif self.hparams.network_architecture == "linear_pseudoinv":
            self.eeg_net = SimpleNetwork(self.forward_model.Nsources*3, self.forward_model.Nsources*out_channel,self.hparams.network_size)
        elif self.hparams.network_architecture == "ConvDip":
            self.eeg_net = ConvDipModel(out_dim=self.forward_model.Nsources * out_channel, info=get_info())
        elif self.hparams.network_architecture == "DeepSIF":
            self.eeg_net = TemporalInverseNet(self.forward_model.Nsens, self.forward_model.Nsources*out_channel, self.hparams.network_size)
        elif self.hparams.network_architecture == "DeepSIF_pseudoinv":
            self.eeg_net = TemporalInverseNet(self.forward_model.Nsources*out_channel, self.forward_model.Nsources * out_channel,
                                              self.hparams.network_size)

        elif self.hparams.network_architecture == "egnn":
            self.eeg_net = EGNN(in_channels, self.hparams.network_size, out_channel, in_edge_nf=9, device=self.device, act_fn=nn.SiLU(), n_layers=self.hparams.depth, residual=True,)
            # We need one set of edges for every forward matrix
            self.coords = torch.cat([torch.tensor(self.forward_model.scs_loc),
                                                   torch.tensor(self.forward_model.sens_loc)],dim=0).float().to(self.device)
            self.EEG_edges = {key: get_edges_EEG(torch.tensor(self.forward_model.scs_loc), torch.tensor(self.forward_model.sens_loc),
                                           self.forward_model.forward_matrices[key]["L"], self.forward_model.forward_matrices[key]["P"])
                            for key in self.forward_model.forward_matrices.keys()}
        elif "transformer" in self.hparams.network_architecture:
            assert self.hparams.network_size % 6 == 0, "We need the network size to be divisible by 6 for current pos embedding"
            if "num_heads" in self.hparams:
                num_heads = self.hparams.num_heads
            else:
                num_heads = 4
            if "decoderhalf" in self.hparams.network_architecture:
                dec_dim = self.hparams.depth //2
            elif "decoderdouble" in self.hparams.network_architecture:
                dec_dim = self.hparams.depth * 2
            elif "num_decoder_layers" in self.hparams and self.hparams.num_decoder_layers is not None:
                dec_dim = self.hparams.num_decoder_layers
            else:
                dec_dim = self.hparams.depth
            self.eeg_net = SimpleTransformer(channels=self.hparams.network_size, num_heads=num_heads,
                                             out_channels=out_channel, num_layers=self.hparams.depth, meas_channels=1,
                                             source_channels=in_channels, source_norm="sourcenorm" in self.hparams.network_architecture,
                                             norm_first="normfirst" in self.hparams.network_architecture, dec_dim=dec_dim)

            if "NoPos" in self.hparams.network_architecture:
                self.eeg_net.set_source_embedding(torch.zeros(self.forward_model.scs_loc.shape).float())
                self.eeg_net.set_meas_embedding(torch.zeros(self.forward_model.sens_loc.shape).float())
            else:
                self.eeg_net.set_source_embedding(torch.tensor(self.forward_model.scs_loc).float())
                self.eeg_net.set_meas_embedding(torch.tensor(self.forward_model.sens_loc).float())
        elif self.hparams.network_architecture == "UnetMeasurementEncoded":
            unet = UNetModel(
                image_size=self.img_size,
                in_channels=in_channels,
                model_channels=self.hparams.network_size,
                out_channels=out_channel,
                num_res_blocks=2,
                attention_resolutions=(4, 8),  # include 8?
                dropout=0,
                channel_mult=channel_multiplier,  # Maybe 1 Layer deeper?
                conv_resample=True,
                dims=dims,
                num_heads=4,  # Increase to 4?
                eeg_measurement_size=self.cond_dim,
                resblock_updown=True,  # Set to True
                encoder_channels=288,
            )
            meas_encoder = MeasurementEncoderTransformer(channels=288, num_heads=4, num_layers=2, meas_channels=1)
            meas_encoder.set_meas_embedding(torch.tensor(self.forward_model.sens_loc).float())
            self.eeg_net = MeasureUNet(unet, meas_encoder)

        elif self.hparams.network_architecture in ["TimeUnet", "SmallTimeUnet"]:
            self.eeg_net = UNetModel(
                image_size=self.img_size,
                in_channels=in_channels,
                model_channels=self.hparams.network_size,
                out_channels=out_channel,
                num_res_blocks=1 if "SmallTimeUnet" == self.hparams.network_architecture else 2,
                attention_resolutions=(4, 8),  # include 8?
                dropout=0,
                channel_mult=channel_multiplier, #Maybe 1 Layer deeper?
                conv_resample=True,
                dims=dims,
                num_heads=4,  # Increase to 4?
                eeg_measurement_size=self.cond_dim,
                resblock_updown=True,  # Set to True
                time_steps=self.time_steps,
            )
        elif self.hparams.network_architecture == "TimeDeepSif":
            self.eeg_net = DeepSIFNet(self.forward_model.Nsens, self.forward_model.Nsources*out_channel, self.hparams.network_size,
                                      time_steps=self.time_steps)
        elif self.hparams.network_architecture in ["SupLasso", "SupLassoPos", "SupLassoPosZero", "SupLassoZero", "SupGammaMap", "SupeLORETA",
                      "SupMNE", "SupdSPM", "SupsLORETA", "Supbeamformer","SupChampagne","SupMNEeLORETA"]:
            self.eeg_net = ClassicalApproach(self.hparams.network_architecture, self.forward_model)
            self.scaling_factor=1.0
            self.automatic_optimization = False
        elif self.hparams.network_architecture =="unet":
            self.eeg_net = UNetModel(
                image_size=self.img_size,
                in_channels=in_channels,
                model_channels=self.hparams.network_size,
                out_channels=out_channel,
                num_res_blocks=2,
                attention_resolutions=(4, 8),  # include 8?
                dropout=0,
                channel_mult=channel_multiplier, #Maybe 1 Layer deeper?
                conv_resample=True,
                dims=dims,
                num_heads=4,  # Increase to 4?
                eeg_measurement_size=self.cond_dim,
                resblock_updown=True,  # Set to True
            )
        else:
            raise ValueError(f"Unknown network architecture {self.hparams.network_architecture}")

        init_weights(self.eeg_net)


        if "loss_type" not in self.hparams:
            warnings.warn("No loss type provided, using l1")
            self.loss = nn.L1Loss(reduction="sum")
        elif self.hparams.loss_type == "l1":
            self.loss = nn.L1Loss(reduction="sum")
        elif self.hparams.loss_type == "l2":
            self.loss = nn.MSELoss(reduction="sum")
        elif self.hparams.loss_type == "huber":
            self.loss = nn.HuberLoss(reduction='sum', delta=1.0)
        elif self.hparams.loss_type == "cosine":
            self.loss = lambda pred, gt:  (1-nn.functional.cosine_similarity(pred.flatten(1), gt.flatten(1), dim=1)).sum()
        elif self.hparams.loss_type == "weighted_mse":
            self.loss = lambda pred, gt: activation_weighted_mse(pred, gt, zero_scale=0.01)
        elif self.hparams.loss_type == "weighted_mae":
            self.loss = lambda pred, gt: activation_weighted_mae(pred, gt, zero_scale=0.01)
        elif self.hparams.loss_type == "dynamicweighted_mse":
            self.loss = lambda pred, gt: activation_weighted_mse(pred, gt, None)
        elif self.hparams.loss_type == "focal_loss":
            self.loss = lambda pred, gt: activation_focal_loss(pred, gt) + nn.functional.l1_loss(pred, gt, reduction="sum")
        elif self.hparams.loss_type == "pseudo_huber":
            # Following the Huber regularizer of Improved techniques for training consistency models
            # ICLR 2023 Y Song, P Dhariwal
            c = 0.00054 * float(self.forward_model.Nsources)**0.5
            self.loss = PseudoHuberLoss(reduction="sum", c=c)
        else:
            raise ValueError(f"Unknown loss type {self.hparams.loss_type}")
        self.rec_loss = nn.MSELoss(reduction="mean")

        if self.hparams.hausdorff_distance_weight:
            self.hausdorff_loss = WeightedHausdorffDistance(self.forward_model.dis_matrix)

    def prepare_input(self, x, cond_variable, forward_index=""):

        pseudo_inv = self.forward_model.pseudo_inv_specific(x, forward_index, mode=self.pseudo_inv_mode)
        if isinstance(self.scaling_factor, torch.Tensor):
            # We have an adaptive scaling that sets the maximal value to 1
            max_v = 1 / torch.clamp(torch.max(pseudo_inv.abs().flatten(1), dim=1)[0], min=1e-8)
            self.scaling_factor = max_v[:, None, None]

        pseudo_inv = pseudo_inv * self.scaling_factor
        return pseudo_inv, cond_variable

    def forward(self, x, cond_variable=None, forward_index=""):
        if self.hparams.network_architecture in ["SupLasso", "SupLassoPos", "SupLassoPosZero", "SupLassoZero", "SupGammaMap",
                                          "SupeLORETA",
                                          "SupMNE", "SupdSPM", "SupsLORETA", "Supbeamformer", "SupChampagne"]:
            # We need to return the output of the classical approach

            out =  self.eeg_net(x, forward_index)
            out = self.forward_model.apply_morph(out)

            return PointDistribution(out)
        pseudo_inv, cond_variable = self.prepare_input(x, cond_variable, forward_index)
        # Problem when pseudo_inv has different amount of channels (fixed orientation vs flexible orientation)
        if pseudo_inv.shape[-1] > self.in_channels:
            pseudo_inv = torch.norm(pseudo_inv, dim=-1, keepdim=True)
        elif pseudo_inv.shape[-1] < self.in_channels:
            pseudo_inv = torch.nn.functional.pad(pseudo_inv, (0, self.in_channels - pseudo_inv.shape[-1]), "constant", 0)

        if self.hparams.network_architecture in ["linear", "linearBN", "DeepSIF", "TimeDeepSif", "ConvDip"]:
            if self.hparams.network_architecture in ["linear", "TimeDeepSif", "ConvDip"]:
                # We need to scale x for stability!
                x = x / 100
            # Or use batchnorm within the network
            pred_y = self.eeg_net(x)
            # If we have some morph, we need to morph the output
            pred_y = self.forward_model.apply_morph(pred_y)

        elif self.hparams.network_architecture in ["linear_pseudoinv","DeepSIF_pseudoinv"]:
            # First we need to extract the linear sources from the pseudo_inv
            pseudo_inv_flat = pseudo_inv.flatten(1)
            pred_y = self.eeg_net(pseudo_inv_flat).reshape(x.shape[0], pseudo_inv.shape[1], -1)
        elif self.hparams.network_architecture == "egnn":
            # TODO Make everything   right for the egnn format!
            edges = self.EEG_edges[forward_index]
            # For simplicity, we iterate through the batches, as otherwise we have to take care of correct edge features?!
            input_egnn = torch.cat([pseudo_inv, x[...,None].expand(-1,-1,3)], dim=1).to(x.device)
            #We stack measurements on top of features!
            pred_y = self.eeg_net(input_egnn, self.coords.to(x.device), edges[0].to(x.device), edges[1].to(x.device))[:,:self.forward_model.Nsources]
        elif "transformer" in self.hparams.network_architecture:
            if len(x.shape) == 2:
                # Adding the single channel dimension
                x = x[..., None]
            if "residual" in self.hparams.network_architecture:
                pred_y = self.eeg_net(x, pseudo_inv, residual=True)
            elif "RandomPseudo" in self.hparams.network_architecture:
                pred_y = self.eeg_net(x, torch.randn_like(pseudo_inv), residual=False)
            else:
                pred_y = self.eeg_net(x, pseudo_inv, residual=False)
        elif self.hparams.network_architecture == "UnetMeasurementEncoded":
            pseudo_inv_volume = self.forward_model.vector_to_volume(pseudo_inv)
            if len(x.shape) == 2:
                # Adding the single channel dimension
                x = x[..., None]

            pred_y = self.eeg_net(x, self.forward_model.add_padding(pseudo_inv_volume),
                              torch.zeros(pseudo_inv_volume.shape[0], device=pseudo_inv_volume.device),
                              cond_variable)
            #Remove Padding
            pred_y = self.forward_model.remove_padding(pred_y)
            pred_y = self.forward_model.volume_to_vector(pred_y)
        else:
            pseudo_inv_volume = self.forward_model.vector_to_volume(pseudo_inv)

            pred_y = self.eeg_net(self.forward_model.add_padding(pseudo_inv_volume),
                          torch.zeros(pseudo_inv_volume.shape[0], device=pseudo_inv_volume.device),
                          cond_variable)
            #Remove Padding
            pred_y = self.forward_model.remove_padding(pred_y)
            pred_y = self.forward_model.volume_to_vector(pred_y)

        if "normal" in self.hparams.probabilistic_out:
            if len(pred_y.shape) == 3:
                mean, log_var = torch.chunk(pred_y, 2, dim=2)
            else:
                mean, log_var = torch.chunk(pred_y, 2, dim=1)
            std = nn.functional.softplus(log_var) + 1e-4
            mean = mean / self.scaling_factor
            out_dist = torch.distributions.Normal(mean, std)
        elif "MOE" in self.hparams.probabilistic_out:
            if len(pred_y.shape) == 3:
                mean, log_var, pi = torch.chunk(pred_y, 3, dim=2)
            else:
                mean, log_var, pi = torch.chunk(pred_y, 3, dim=1)
            std = nn.functional.softplus(log_var) + 1e-4
            mean = mean / self.scaling_factor
            Mixture = torch.distributions.categorical.Categorical(logits=move_expert_channel(pi, self.n_experts))
            Component = torch.distributions.Normal(
                move_expert_channel(mean, self.n_experts),
                move_expert_channel(std, self.n_experts),
            )
            out_dist = MixtureOfExperts(Mixture, Component)
        elif "residue_point" in self.hparams.probabilistic_out:
            out_dist = PointDistribution((pseudo_inv-pred_y) / self.scaling_factor)
        else:
            out_dist = PointDistribution(pred_y / self.scaling_factor)

        return out_dist

    def training_step(self, batch, batch_idx):
        return self.compute_losses(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            self.compute_losses(batch, mode="valid", plot_volumes=batch_idx == 1)

    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            self.compute_losses(batch, mode="test")

    def get_cond_variable(self, batch, forward_index=""):
        cond_variable = None
        return cond_variable


    def compute_losses(self, batch, mode="train", plot_volumes=False):
        target = batch["sources"]
        forward_index = np.random.choice(list(self.forward_model.forward_matrices.keys()))
        if self.hparams.noise_free == "add_noise":
            sensors = self.forward_model.compute_noisy_forward(target)
        elif self.hparams.noise_free in ["multiple_forward", "multiple_lead"]:
            sensors = self.forward_model.compute_noisy_forward(target, forward_index=forward_index)
        elif self.hparams.noise_free:
            sensors = self.forward_model.forward_specific(target,forward_index=forward_index)
        else:
            sensors = batch["sensors"]
            forward_index = batch["forward_index"]

        batch["sensors"] = sensors
        cond_variable = self.get_cond_variable(batch, forward_index=forward_index)
        y_hat_dist = self.forward(sensors, cond_variable, forward_index=forward_index)



        if self.hparams.hausdorff_distance_weight:
            assert "source_centers" in batch.keys(), "Missing source centers, can't compute hausdorff loss!"
            # We could also use the mixture coefficient of a MOE model for hausdorff Distance?
            if "MOE" in self.hparams.probabilistic_out:
                flatten_pred = y_hat_dist.mixture_distribution.probs[...,-1]
            else:
                flatten_pred = y_hat_dist.mean
            centers = [center[:n] for center, n in zip(batch["source_centers"], batch["Nsources"])]
            hausdorff_loss = self.hausdorff_loss(flatten_pred, centers) * self.hparams.hausdorff_distance_weight
        else:
            hausdorff_loss = torch.tensor([0.0], device=target.device)

        # By adding back the scaling factor, we are able to not change the learning rate accross datasets?
        loss = self.loss(y_hat_dist.mean*self.scaling_factor, target*self.scaling_factor) / self.forward_model.Nsources
        rec_loss = self.rec_loss(self.forward_model.forward_specific(y_hat_dist.mean, forward_index), batch["sensors"])
        # Noise free alternative of reconstruction loss
        #rec_loss = self.rec_loss(self.forward_model.forward_specific(y_hat_dist.mean, forward_index), self.forward_model.forward_specific(target, forward_index))
        log_likelihood_loss = self.log_likelihood_loss(y_hat_dist, target, forward_index)

        additional_losses = compute_metrics(target, y_hat_dist.mean,measurements=batch["sensors"],
                                            forward_model=self.forward_model, compute_emd=False)

        additional_losses = {f"{mode}_" + k: v for k, v in additional_losses.items()}

        # Logging

        self.log(f"{mode}_reconstruction_error", rec_loss.detach().cpu())
        self.log(f"{mode}_loss", loss.detach().cpu() / target.shape[0])
        self.log(f"{mode}_hausdorff_loss", hausdorff_loss.detach().cpu())
        self.log(
            f"{mode}_NMSE",
            (loss.detach().cpu() / (torch.sum(target.detach().cpu() ** 2) / self.forward_model.Nsources)).detach(),
        )
        self.log(f"{mode}_loglike", log_likelihood_loss.detach().cpu() / target.shape[0])
        self.log_losses_dict(additional_losses)

        if plot_volumes:
            self.plot_prediction(sensors, target, y_hat_dist, f"{mode}-Prediction-Step{self.global_step}",
                                 forward_index=forward_index, cond_variable=cond_variable)

        return (loss + self.hparams.lambda_rec * rec_loss + self.hparams.lambda_like * log_likelihood_loss \
            + hausdorff_loss)


    def log_likelihood_loss(self, dist, target, forward_index):
        return (
            torch.sum(-1 * dist.log_prob(target))
            / self.forward_model.Nsources
        ) / target.shape[0]

    def on_validation_epoch_end(self):
        # If the selected scheduler is a ReduceLROnPlateau scheduler.
        if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.lr_scheduler.step(self.trainer.callback_metrics["valid_loss"])
        else:
            self.lr_scheduler.step()
        return

    def configure_optimizers(self):
        if "learned" in self.hparams.pseudo_inv_mode:
            #TODO maybe remove later, just for testing
            optimizer = AdamW([
                {"params": self.eeg_net.parameters(), "lr": self.hparams.learning_rate},
                {"params": self.forward_model.parameters(), "lr": self.hparams.learning_rate*0.001},
            ], lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        else:
            optimizer = AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        self.lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=5, factor=0.5, verbose=True)
        return optimizer

    def log_losses_dict(self, losses):
        for k, v in losses.items():
            if isinstance(v, torch.Tensor):
                self.log(k, v.mean().detach().cpu())
            elif isinstance(v, np.ndarray):
                self.log(k, v.mean())
            else:
                self.log(k, v)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = super(SupervisedEEGModel, SupervisedEEGModel).add_model_specific_args(parent_parser)
        parser.add_argument("--network_size", default=32, type=int)
        parser.add_argument("--lambda_rec", default=0.0, type=float)
        parser.add_argument("--lambda_like", default=0.0, type=float)
        parser.add_argument("--noise_free", default="multiple_lead", type=str)
        parser.add_argument("--probabilistic_out", default="point", type=str)
        parser.add_argument("--source_reg", default=0.0, type=float)
        parser.add_argument("--hausdorff_distance_weight", default=0.0, type=float, help="weight of the hausdorff loss")

        return parser

    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams)

    def on_train_end(self):
        if "HBN" in self.hparams.data_set:
            self.forward_model = EEGforward("HBNall")

        #self.evaluate_model(False)

    def tune_prediction_pytorch(self, sensor, pred_source, forward_index):
        self.forward_model.forward_noise = False
        pred_eeg = self.forward_model.forward_specific(pred_source, forward_index)
        start_scale = torch.sum(sensor.flatten(1) * pred_eeg.flatten(1), axis=1) / torch.sum(pred_eeg**2, axis=1)
        return pred_source * (start_scale[:, None, None])

    def evaluate_model(self, noise_free=False, forward_index = None, plot_add="", data_leadfield_index=None, max_length=200, recompute_cov=False, prediction_path="",
                       pred_pseudo=False, sensor_subset=None, compute_emd=False, plot_pred=False):
        if data_leadfield_index is None:
            data_leadfield_index = forward_index
        val_loader = self.val_dataloader(forward_index=data_leadfield_index, max_length=max_length)
        model_time = 0
        pseudo_time = 0
        total_loss = None
        if recompute_cov:
            if isinstance(val_loader.dataset, torch.utils.data.Subset):
                self.forward_model.set_mne_obj(torch.tensor(val_loader.dataset.dataset.data["sensors"]))
            else:
                # We recompute the covariance matrix for the evaluation on the full dataset
                self.forward_model.set_mne_obj(torch.tensor(val_loader.dataset.data["sensors"]))

        if prediction_path:
            result_pred = []

        for batch in tqdm(val_loader):
            with torch.no_grad():
                if batch["sources"].device != self.device:
                    # a little bit ugly, as source centers is a list of tensors!
                    batch = {k: v.to(self.device) if k != "source_centers" else [c.to(self.device) for c in v] for k, v in batch.items()}
                if noise_free:
                    batch["sensors"] = self.forward_model.forward_specific(batch["sources"], forward_index=data_leadfield_index)

                start_time = time.time()
                pseudo_out = self.forward_model.pseudo_inv_specific(batch["sensors"], forward_index=forward_index)
                pseudo_time += time.time() - start_time
                cond_variable = self.get_cond_variable(batch, forward_index=forward_index)
                start_time = time.time()
                out_dist = self.forward(batch["sensors"], cond_variable, forward_index=forward_index)
                model_time += time.time() - start_time
            if isinstance(out_dist, torch.distributions.Distribution):
                out_dist = out_dist.mean
            target = batch["sources"]

            # Remove Padding when necessary
            if self.forward_model.padding:
                target = self.forward_model.remove_padding(target)
                out_dist = self.forward_model.remove_padding(out_dist)
                pseudo_out = self.forward_model.remove_padding(pseudo_out)

            #We try to scale the prediction based on the sensor values
            #start_time = time.time()
            #tuned_pred = self.tune_prediction_pytorch(batch["sensors"], out_dist, forward_index)
            #tune_time += time.time() - start_time
            if target.shape[-1] < out_dist.shape[-1]:
                # We have a fixed orientation target:
                # However the results could also be negative, so we take the first orientation!
                out_dist = out_dist[..., :target.shape[-1]]
                #tuned_pred = tuned_pred[..., :target.shape[-1]]
            elif target.shape[-1] > out_dist.shape[-1]:
                # We have a fixed orientation prediction
                target = target[..., :out_dist.shape[-1]]
                pseudo_out = pseudo_out[..., :out_dist.shape[-1]]
                #tuned_pred = tuned_pred[..., :out_dist.shape[-1]]

            if prediction_path:
                result_pred.append(out_dist.cpu().numpy())

            losses = compute_metrics(target, out_dist, forward_model=self.forward_model, forward_index=forward_index,measurements=batch["sensors"], compute_emd=compute_emd)
            if pred_pseudo:
                pseudo_losses = compute_metrics(target, pseudo_out, forward_model=self.forward_model,
                                            forward_index=forward_index,measurements=batch["sensors"],compute_emd=compute_emd)
            #tuned_losses = compute_metrics(target, tuned_pred, forward_model=self.forward_model, forward_index=forward_index)

            if total_loss is not None:
                total_loss = {k: torch.cat([total_loss[k], v], dim=0) for k, v in losses.items()}
                if pred_pseudo:
                    pseudo_loss = {k: torch.cat([pseudo_loss[k], v], dim=0) for k, v in pseudo_losses.items()}
                # tuned_loss = {k: torch.cat([tuned_loss[k] + v], dim=0) for k, v in tuned_losses.items()}
            else:
                # Total Loss empty
                total_loss = {k: v for k, v in losses.items()}
                if pred_pseudo:
                    pseudo_loss = {k: v for k, v in pseudo_losses.items()}
                # tuned_loss = {k: v for k, v in tuned_losses.items()}

        def compute_mean_std(loss_dict):
            loss_std = {k: torch.std(v).item() for k, v in loss_dict.items()}
            loss_mean = {k: torch.mean(v).item() for k, v in loss_dict.items()}
            loss_mean.update({k + "_std": v for k, v in loss_std.items()})
            return loss_mean

        losses_mean = compute_mean_std(total_loss)
        losses_mean.update({"runtime": model_time})
        # tuned_losses_mean = compute_mean_std(tuned_loss)
        # tuned_losses_mean.update({"runtime": tune_time})
        loss_df = pd.DataFrame([losses_mean],  # , tuned_losses_mean],
                               index=[f"{self.hparams.experiment_name}nf{noise_free}{plot_add}",
                                      # f"Tuned{self.hparams.experiment_name}nf{noise_free}{plot_add}"
                                      ])

        if pred_pseudo:
            pseudo_losses_mean = compute_mean_std(pseudo_loss)
            pseudo_losses_mean.update({"runtime": pseudo_time})
            loss_df[f"Pseudonf{noise_free}{plot_add}"] = pd.Series(pseudo_losses_mean)

        loss_df.to_csv(f"{self.log_dir}/Evaluation-results-{plot_add}NF{noise_free}.csv")

        if plot_pred:
            self.plot_prediction(
                batch["sensors"],
                target,
                out_dist,
                plot_name=f"Final-Evaluation-NF{noise_free}{plot_add}",
                log_dir=f"lightning_logs/{self.hparams.experiment_name}",  # self.log_dir,
                cond_variable=cond_variable,
                forward_index=forward_index,
            )

        if prediction_path:
            result_pred = np.concatenate(result_pred, axis=0)
            save_compressed_pickle(prediction_path, result_pred)

        return loss_df

    def set_time_step(self, time_step):
        """
        Sets the timestep to all submodules
        """

        def set_time_module(module):
            if hasattr(module, 'time_steps'):
                module.time_steps = time_step
        self.eeg_net.apply(set_time_module)

def cli_main():
    # ------------
    # args
    # ------------
    parser = ArgumentParser()

    #parser = pl.Trainer.add_argparse_args(parser)
    # Not sure which parser are now not working anymore
    parser = add_lightning_args(parser)
    parser = SupervisedEEGModel.add_model_specific_args(parser)

    args = parser.parse_args()
    pl.seed_everything(args.seed)
    # ------------
    # model
    # ------------
    model = SupervisedEEGModel(args)

    # ------------
    # training
    # ------------cd
    os.makedirs(f"lightning_logs/{args.experiment_name}", exist_ok=True)

    try:
        from pytorch_lightning.loggers import WandbLogger

        # initialise the wandb logger and name your wandb project
        logger = WandbLogger(name=args.experiment_name, save_dir=f"lightning_logs/{args.experiment_name}", project='EEG_Project')
        logger.watch(model.eeg_net, log_freq=250)
    except ImportError as e:
        print("Probably Wandb is not installed")
        logger = pl_loggers.TensorBoardLogger("./", version=args.experiment_name)


    if "accumulate_grad_batches" in args:
        accumulate_grad_batches = args.accumulate_grad_batches
    else:
        accumulate_grad_batches = 1
    #trainer = pl.Trainer.from_argparse_args(args, logger=tensorboard)
    trainer = pl.Trainer(max_epochs=args.max_epochs, accelerator=args.accelerator, devices=args.devices, logger=logger,
                         check_val_every_n_epoch=5, accumulate_grad_batches=accumulate_grad_batches, default_root_dir=f"lightning_logs/{args.experiment_name}",
                         precision=args.precision
                         )
    trainer.progress_bar_callback._refresh_rate = 100
    if "marco" not in os.getcwd() and args.auto_scale_batch_size:
        trainer.tune(
            model,
            scale_batch_size_kwargs={
                "mode": args.auto_scale_batch_size,
                "max_trials": 3,
                "init_val": args.batch_size,
            },
        )
    if args.resume_from_checkpoint is not None:
        trainer.fit(model, ckpt_path=args.resume_from_checkpoint)
    else:
        trainer.fit(model)

    trainer.save_checkpoint(f"lightning_logs/{args.experiment_name}/checkpoints/final_model.ckpt")
    # ------------
    # testing
    # ------------
    trainer.test(dataloaders=model.test_dataloader())


if __name__ == "__main__":
    cli_main()
