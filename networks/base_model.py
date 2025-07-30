import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset
from utils.data_utils import EEGDataset, custumn_collate

from argparse import ArgumentParser, ArgumentTypeError
from utils.esinet_data import EsinetDataset
import torch
import traceback
from distributions.MixtureOfExperts import MixtureOfExperts
from distributions.pointdistribution import PointDistribution
from utils.plotting_utils import plot_animation_volumes, plot_mnist_overview
class BaseEEGModel(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        """
        if kwargs.get("data_set", "eeg") == "eeg":
            self.forward_model = EEGforward(padding=False)
        elif kwargs.get("data_set", "eeg") == "cifar10":
            self.forward_model = Blurforward(img_shape=(32, 32))
        else:
            self.forward_model = Blurforward()
        """
    # ------------
    # data
    # ------------

    def get_data_loader(self, mode="train", forward_index=None, max_length=None):
        if self.hparams.data_set == "eeg":
            dataset = EEGDataset(mode=mode)
        elif "generate" in self.hparams.data_set:
            if "validation" == mode and forward_index is not None:
                mode = self.hparams.data_set.replace("generate", f"validation{forward_index}")
            else:
                mode = self.hparams.data_set
            dataset = EEGDataset(mode=mode, forward_model=self.forward_model)
        elif "esinet" in self.hparams.data_set:
            dataset = EsinetDataset(
                mode=mode, oneD="oneD" in self.hparams.data_set, sampling = "ico3" if "ico3" in self.hparams.data_set else "ico4",
                normalize="normalize" in self.hparams.data_set,
            )
        else:
            raise ValueError(f"unknown Dataset {self.hparams.data_set}")
        if max_length is not None:
            dataset = Subset(dataset, indices=range(min(max_length, len(dataset))))
        return DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=mode == "train", num_workers=4, collate_fn=custumn_collate)

    def train_dataloader(self):
        return self.get_data_loader("train")

    def val_dataloader(self, forward_index=None, max_length=None):
        return self.get_data_loader("validation", forward_index, max_length=max_length)

    def test_dataloader(self):
        return self.get_data_loader("test")

    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--learning_rate", type=float, default=0.00001)
        parser.add_argument("--weight_decay", type=float, default=0.0)
        parser.add_argument("--include_measurements", default=False, action="store_true")  # Depricated
        parser.add_argument("--experiment_name", default="", type=str)
        parser.add_argument("--data_set", default="MNEVolgenerate_N0-30_S1-20_E0.005-0.04", type=str)
        parser.add_argument("--use_noise", default=False, action="store_true")
        parser.add_argument("--depth", default=3, type=int)
        parser.add_argument("--seed", default=1, type=int)
        parser.add_argument("--batch_size", default=32, type=int)
        parser.add_argument("--network_architecture", default="unet", type=str)
        parser.add_argument("--num_heads", default=4, type=int)
        parser.add_argument("--num_decoder_layers", default=None, type=int)
        parser.add_argument("--accumulate_grad_batches", default=1, type=int)
        parser.add_argument("--loss_type", default="l1", type=str)
        parser.add_argument("--min_snr", default=0.0, type=float)
        parser.add_argument("--max_snr", default=20.0, type=float)
        parser.add_argument("--correlated_noise", default=0.0, type=float)
        parser.add_argument("--forward_noise", default=0.0, type=float)
        parser.add_argument("--pseudo_inv_mode", default="loaded", type=str)
        parser.add_argument("--precision", default=32, type=lambda x: int(x) if x.isdigit() else x)
        return parser

    def plot_prediction(self, sensors, target, pred, plot_name="", log_dir=None, forward_index="", cond_variable=None):
        if log_dir is None:
            log_dir = self.log_dir
        if isinstance(pred, MixtureOfExperts):
            pred_cat = self.forward_model.vector_to_volume(pred.mixture_distribution.probs[..., -1])
            pred_experts = [self.forward_model.vector_to_volume(pred.component_distribution.mean[..., i]) for i in range(pred.n_experts)]
        else:
            pred_cat = None
        if isinstance(pred, torch.distributions.Distribution):
            if isinstance(pred, PointDistribution):
                pred_stddev = None
            else:
                pred_stddev = pred.stddev

            pred = pred.mean
        else:
            pred_stddev = None
        if "eeg" in self.hparams.data_set or "esinet" in self.hparams.data_set or "MNE" in self.hparams.data_set:
            if len(target.shape) <4:
                target = self.forward_model.vector_to_volume(target)
                pred = self.forward_model.vector_to_volume(pred)

                pred_stddev = self.forward_model.vector_to_volume(pred_stddev) if pred_stddev is not None else None
                if cond_variable is not None:
                    cond_variable = self.forward_model.vector_to_volume(cond_variable)
            peudo_inv = self.forward_model.pseudo_inv_specific(sensors, forward_index, return_volume=True, mode=self.pseudo_inv_mode)
            volumes = [
                target[0].norm(dim=0).detach().cpu().float().numpy(),
                peudo_inv[0].norm(dim=0).detach().cpu().float().numpy(),
                pred[0].norm(dim=0).detach().cpu().float().numpy(),
            ]
            plot_names = [
                "Ground Truth",
                "Pseudo Inverse",
                "Predicted",
            ]
            if (pred_stddev is not None) and ("normal" in self.hparams.probabilistic_out):
                volumes.append((pred_stddev[0].norm(dim=0)).detach().cpu().float().numpy())
                plot_names.append("Pred StdDev")
            if cond_variable is not None:
                volumes.append(cond_variable[0].norm(dim=0).detach().cpu().float().numpy())
                plot_names.append("Conditioning Variable")
            try:
                plot_animation_volumes(
                    volumes,
                    filename=f"{log_dir}/{plot_name}.mp4" if plot_name else None,
                    names=plot_names,
                )
            except:
                print("Could not plot")
                traceback.print_exc()


            # Also do a 2d Plot of maximal activated region
            # Find max region:
            max_slice_idx = target[0].norm(dim=0).flatten(1).sum(dim=1).argsort(dim=0).detach().cpu().numpy()[::-1]
            target[0].norm(dim=0)

            imgs = [
                target[0].norm(dim=0).detach().cpu().float().numpy()[max_slice_idx[:4]],
                peudo_inv[0].norm(dim=0).detach().cpu().float().numpy()[max_slice_idx[:4]],
                pred[0].norm(dim=0).detach().cpu().float().numpy()[max_slice_idx[:4]],
            ]
            plot_names = ["Target", "PseudoInv", "Prediction"]
            if pred_stddev is not None and self.hparams.probabilistic_out != "point":
                if len(pred_stddev.shape) == 4:
                    #Adding a dimension for the channel
                    pred_stddev = pred_stddev[:,None]
                imgs.append(pred_stddev[0].norm(dim=0).detach().cpu().float().numpy()[max_slice_idx[:4]])
                plot_names.append("Pred StdDev")
            if cond_variable is not None:
                imgs.append(cond_variable[0].norm(dim=0).detach().cpu().float().numpy()[max_slice_idx[:4]])
                plot_names.append("Conditioning Variable")
            if pred_cat is not None:
                imgs.append(pred_cat[0].mean(dim=0).detach().cpu().float().numpy()[max_slice_idx[:4]])
                plot_names.append("Pred Mixture")
                for i, x in enumerate(pred_experts):
                    imgs.append(x[0].norm(dim=0).detach().cpu().float().numpy()[max_slice_idx[:4]])
                    plot_names.append(f"Expert {i}")

            try:
                plot_mnist_overview(
                    imgs,
                    filename=f"{log_dir}/Flat{plot_name}.png",
                    y_names=plot_names,
                    names=[f"Slice {x}" for x in max_slice_idx[:4]]
                )

            except:
                print("Could not plot")
                traceback.print_exc()

        else:
            try:
                # Plot 2D Mnist?
                imgs = [
                    sensors[:10].detach().cpu().numpy().transpose(0, 2, 3, 1).clip(0, 1),
                    target[:10].detach().cpu().numpy().transpose(0, 2, 3, 1).clip(0, 1),
                    pred[:10].detach().cpu().numpy().transpose(0, 2, 3, 1).clip(0, 1),
                ]
                # We need to optionally squeeze the last dimension
                imgs = [x.squeeze(-1) if x.shape[-1] == 1 else x for x in imgs]
                plot_mnist_overview(
                    imgs,
                    filename=f"{log_dir}/{plot_name}.png",
                    y_names=["Input", "Target", "Prediction"],
                )
            except:
                print("Could not plot, probably in tuning")


def float_or_list(value):
    # Check if the value is a single float
    if "," not in value:
        try:
            return float(value)
        except ValueError:
            raise ArgumentTypeError(f"Invalid float value: {value}")

    # Split the value by commas and convert each part to a float
    parts = value.split(",")
    try:
        return [float(part) for part in parts]
    except ValueError:
        raise ArgumentTypeError(f"Invalid list value: {value}")
