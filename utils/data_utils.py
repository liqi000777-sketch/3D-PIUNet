from torch.utils.data import Dataset
import numpy as np
import scipy.io
import torch
from scipy.stats import norm
import re
import colorednoise as cn

import gzip
import pickle

class EEGDataset(Dataset):
    """EEG dataset."""

    def __init__(self, root_dir="data", mode="train", transform=None, forward_model=None, one_D=True):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
        self.forward_model = forward_model
        assert one_D, "Saving data as volume is depricated"
        # TODO check if numpy version exist!
        try:
            if "generate" in mode:
                self.data = None
                self.N = 10000
            else:
                self.data = load_compressed_pickle(root_dir + f"/{mode}fulldata1D")
                #self.data = np.load(root_dir + f"/{mode}fulldata1D.npz", allow_pickle=True)["data"]
                #self.data = np.load(root_dir + f"/{mode}fulldata1D.npy", allow_pickle=True)[()]
                self.N = self.data["sensors"].shape[0]
        except FileNotFoundError:
            print("Packed data was not found, need to extract again ")
            if mode in ["train", "validation", "test"]:
                self.format_individual_data()
            else:
                self.generate_valid_set()



    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if "generate" in self.mode:
            sample = self.generate_random_data(idx)
        else:
            if "forward_index" in self.data.keys():
                forward_index = self.data["forward_index"][idx]
            else:
                forward_index = "base"



            sample = {
                "sensors": self.data["sensors"][idx],
                "sources": self.data["sources"][idx],
                "Nsources": self.data["Nsources"][idx],
                "forward_index": forward_index
            }
            if "source_centers" in self.data.keys():
                sample["source_centers"] = self.data["source_centers"][idx]

        # Padding?
        if self.transform:
            sample = self.transform(sample)

        return sample


    def format_individual_data(self):

        eeg_utils = scipy.io.loadmat(self.root_dir + "/simvars_3D.mat")
        Ngridvox = eeg_utils["Ngridvox"][0, 0]
        Nsens = eeg_utils["Nsens"][0, 0]
        Ntime = eeg_utils["Ntime"][0, 0]
        assert Ntime == 1, "Currently do not implement dynamical time"
        source_inds = eeg_utils["source_inds"][:, 0] - 1
        Ngridvox_xyz = eeg_utils["Ngridvox_xyz"][0]

        if self.mode == "train":
            self.N = int(eeg_utils["Ntrain"][0, 0] * 0.8)
            data_folder = "/train/"
            start_idx = 1
        elif self.mode == "validation":
            self.N = int(eeg_utils["Ntrain"][0, 0] * 0.2)
            data_folder = "/train/"
            start_idx = int(eeg_utils["Ntrain"][0, 0] * 0.8) + 1
        elif self.mode == "test":
            self.N = eeg_utils["Ntest"][0, 0]
            start_idx = 1
            data_folder = "/test/"

        self.data = {
            "sensors": np.zeros((self.N, Nsens), dtype=np.float32),
            "sources": np.zeros((self.N, len(source_inds), 3), dtype=np.float32),
            "Nsources": np.zeros(self.N, dtype=np.float32),
        }
        for i in range(self.N):
            mat = scipy.io.loadmat(self.root_dir + data_folder + f"rep{i + start_idx}.mat")
            self.data["sources"][i] = mat["sources"]
            self.data["sensors"][i] = mat["sensors"][:, 0]
            self.data["Nsources"][i] = mat["nsources"][0, 0]

        save_compressed_pickle(self.root_dir + f"/{self.mode}fulldata1D", self.data)
        #np.savez_compressed(self.root_dir + f"/{self.mode}fulldata1D.npz", data=self.data, pickle_kwargs={'protocol': 4})
        #np.save(self.root_dir + f"/{self.mode}fulldata1D.npy", self.data)


    def generate_random_data(self, idx, forward_index=None):
        # We generate Live data
        # Assume a string that encodes the parameter:
        # generate_Na-b_Sx-y_Eu-v
        # N a-b Noise Level
        # S x-y number of Sources
        # E u-v standard deviation of sources
        assert self.forward_model is not None, "A forward model is necessary for generating data!"
        # TODO Make this more robust?! Also maybe save in the beginning to compute only once
        Rest, N, S, E = self.mode.split("_")
        min_snr,max_snr = map(int, N.replace("N","").split("-"))
        min_scs, max_scs = map(int, S.replace("S", "").split("-"))
        min_std, max_std = map(float, E.replace("E", "").split("-"))
        if "TimeFreq" in Rest:
            self.time_steps = int(re.search(r'TimeFreq([0-9]+)', Rest).group(1))
        else:
            self.time_steps = 1

        if "ForNoise" in Rest:
            #we extract the floating number behind the string using re
            forward_noise = float(re.search(r'ForNoise([0-9]*\.[0-9]+)', Rest).group(1))
        else:
            forward_noise = False
        if "CorrNoise" in Rest:
            correlated_noise = float(re.search(r'CorrNoise([0-9]*\.[0-9]+)', Rest).group(1))
        else:
            correlated_noise = False
        if forward_noise or correlated_noise:
            self.forward_model.set_noise_pattern(correlated_noise, forward_noise)

        self.forward_model.set_snr(min_snr, max_snr)
        if min_scs == 0:
            # Oversampling single Source Setting ~33% of the time
            NSources = max(np.random.randint(int(-0.5*max_scs), max_scs + 1, 1)[0], 1)
        else:
            NSources = np.random.randint(min_scs, max_scs+1, 1)[0]
        source_centers = np.random.randint(0, len(self.forward_model.source_inds), NSources)
        std_sources = np.random.rand(NSources)*(max_std-min_std)+min_std
        sources = []
        for i in range(NSources):
            loc = self.forward_model.scs_loc[source_centers[i]]

            # Calculate pairwise distance from source to all voxel locations
            distances = np.linalg.norm(loc[None] - self.forward_model.scs_loc, axis=1)
            # Use this distance as a basis for a 1d gaussian distribution
            source_amp = norm.pdf(distances, loc=0, scale=std_sources[i])
            # Divide the resulting values by the frobenius norm (per source)
            source_amp = source_amp / np.linalg.norm(source_amp)

            # Calculate random orientation (unit norm)
            orientation = np.random.randn(3)
            orientation /= np.linalg.norm(orientation)

            # multiply orientation by source activation to set values
            source_amp_3D = np.outer(source_amp, orientation)
            sources.append(source_amp_3D)

        sources = np.stack(sources, axis=-1)
        # Map the Nsources to NTimes(1) via the matrix randn(nsources, Ntime) and matrix multiplication?
        # This is probably some kind of mixing?
        if self.time_steps > 1:
            source_timeseries = generate_time_noise(NSources, self.time_steps)
            # Sources are Mixed according to time Signal and time is moved to first dimension
            timed_sources = np.einsum("LOS, ST -> TLO", sources, source_timeseries)

            # raise NotImplementedError("We cannot deal with the extra time dimension yet!")
        else:
            source_timeseries = np.random.normal(size=(NSources, 1))
            timed_sources = np.einsum("LOS, ST -> TLO", sources, source_timeseries)


        sources_grid = torch.Tensor(timed_sources)
        # This will allow for different forward indices within one batch!
        if forward_index is None:
            forward_index = self.forward_model.instances[idx % len(self.forward_model.instances)]
        #forward_index = np.random.choice(self.forward_model.forward_matrices.keys())
        sensors = self.forward_model.compute_noisy_forward(sources_grid, forward_index=forward_index)
        sample = {
            "sensors": sensors,
            "sources": sources_grid,
            "Nsources": NSources,
            "forward_index": forward_index,
            "source_centers": source_centers,
        }
        return sample

    def generate_valid_set(self):
        np.random.seed(42)
        data = []
        if re.search(r'validation([0-9])', self.mode):
            forward_index = int(re.search(r'validation([0-9])', self.mode).group(1))
        else:
            forward_index = None

        for i in range(2000):
            data.append(self.generate_random_data(i, forward_index))
        self.N = 2000
        self.data = {}
        for k in data[0].keys():
            if k == "source_centers":
                self.data[k] = [x[k] for x in data]
            else:
                self.data[k] = np.stack([x[k] for x in data], axis=0)
        # Data for time becomes very large, so we save compressed:
        save_compressed_pickle(self.root_dir + f"/{self.mode}fulldata1D", self.data)
        #np.savez_compressed(self.root_dir + f"/{self.mode}fulldata1D.npz", data=self.data, pickle_kwargs={'protocol': 4})
        #np.save(self.root_dir + f"/{self.mode}fulldata1D.npy", self.data)


def pad_sources(x, target_dim=24):
    _, W, H, D = x["sources"].shape
    x["sources"] = np.pad(x["sources"], ((0, 0), (0, target_dim - W), (0, target_dim - H), (0, target_dim - D)))
    return x


def custumn_collate(batch):
    if "source_centers" in batch[0].keys():
        # We need to use a list instead of a tensor here
        centers = [torch.tensor(x.pop("source_centers"), dtype=torch.long) for x in batch]
        main = torch.utils.data.dataloader.default_collate(batch)
        out =  {**main, "source_centers": centers}
    else:
        out =  torch.utils.data.dataloader.default_collate(batch)

    # When we have time, we add the time dimension to the batchsize
    if "sources" in out.keys() and len(out["sources"].shape) == 4:
        out["sources"] = out["sources"].flatten(0,1)
    if "sensors" in out.keys() and len(out["sensors"].shape) == 3:
        out["sensors"] = out["sensors"].flatten(0,1)
    return out

def generate_time_noise(n_sources, num_timesteps=1000, max_beta=5):
        signals = []
        for i in range(n_sources):
            beta = np.random.rand() * max_beta
            signal = cn.powerlaw_psd_gaussian(beta, num_timesteps)
            signals.append(signal)
        return np.array(signals)


def load_compressed_pickle(filepath):
    with gzip.open(f"{filepath}.pkl.gz", 'rb') as f:
        return pickle.load(f)

def save_compressed_pickle(filepath, data, protocol=4):
    with gzip.open(f"{filepath}.pkl.gz", 'wb') as f:
        pickle.dump(data, f, protocol=protocol)
