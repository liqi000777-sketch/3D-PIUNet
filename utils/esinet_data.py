import numpy as np
from esinet import Simulation
from esinet.forward import create_forward_model, get_info
from torch.utils.data import Dataset
from conditioning.utils import mkfilt_eloreta_v2
import torch
import random
from scipy.spatial import cKDTree
DEFAULT_SETTINGS = {
    'method': 'standard',
    'number_of_sources': (1, 25),
    'extents': (1, 50),  # in millimeters
    'amplitudes': (1e-3, 100),
    'shapes': 'mixed',
    'duration_of_trial': 1.0,
    'sample_frequency': 100,
    'target_snr': (1, 20),
    'beta': (0, 5),
    'beta_noise': (0, 5),
    'beta_source': (0, 5),
    'source_spread': "mixed",
    'source_number_weighting': True,
    'source_time_course': "random",
}


def get_esinet_simulation(sampling="ico4", sfreq=1, settings=None):
    # Create generic Forward Model
    info = get_info(sfreq=sfreq)
    fwd = create_forward_model(info=info, sampling=sampling, fixed_ori=True)

    # Simulate M/EEG data
    if settings is None:
        settings = DEFAULT_SETTINGS.copy()
    settings["sample_frequency"] = sfreq
    sim = Simulation(fwd, info, settings=settings, verbose=1)

    return sim, fwd

def get_esinet_data():
    sim, fwd = get_esinet_simulation()

    n_samples = 5000
    # Seed for train
    #np.random.seed(93)
    #random.seed(93)
    np.random.seed(53)
    random.seed(53)

    train_data = sim.simulate(n_samples=n_samples)
    full_eeg_data = {"measure": np.zeros((n_samples, 61)),
                     "sources": np.zeros((n_samples, 5124))}
    print(train_data.eeg_data[0].get_data().shape, train_data.source_data[0].data.shape)

    for i in range(n_samples):
        full_eeg_data["measure"][i] = train_data.eeg_data[i].get_data()[0,:,0]
        full_eeg_data["sources"][i] = train_data.source_data[i].data[:, 0]


    print("start to save")
    np.save("testesinetdata.npy", full_eeg_data)
    print("Saving completed")

class Transfer_Class():
    def __init__(self, scs_grid, scs_spread):
        self.scs_grid = scs_grid
        self.scs_spread = scs_spread
        self.grid_tree = cKDTree(scs_grid)

        self.distances, self.indices = self.grid_tree.query(scs_spread)
        self.weights = np.zeros((len(scs_grid), len(scs_spread)))
        self.mask_grid = np.zeros((len(scs_grid)))
        for i in range(len(scs_grid)):
            relevant_indices = np.where(self.indices == i)[0]
            if len(relevant_indices) > 0:
                self.mask_grid[i] = len(relevant_indices)
                weight = 1.0 / self.distances[relevant_indices]
                weight /= np.sum(weight)  # Normalize weights
                self.weights[i, relevant_indices] = weight

        self.weight_torch = torch.from_numpy(self.weights).float()
        dists, idxs = self.grid_tree.query(scs_spread, k=8)
        self.inv_weights = np.zeros((len(scs_spread), len(scs_grid)))
        self.not_full = np.zeros(len(scs_spread))
        for j in range(len(scs_spread)):
            valid_idx = np.where(self.mask_grid[idxs[j]] != 0)
            idx = idxs[j, valid_idx]
            dist = dists[j, valid_idx]
            if np.sum(valid_idx) != 8:
                self.not_full[j] = 1
            # TODO I want to exclude points from idx where self.mask_grid[idx] =0
            weight = 1. / dist
            weight /= np.sum(weight)
            self.inv_weights[j, idx] = weight

    def grid_to_spread_complex(self, activation_grid):
        if len(activation_grid.shape) == 1:
            return np.einsum("ji, i -> j", self.inv_weights, activation_grid)
        elif len(activation_grid.shape) == 2:
            # Batched activations!
            return np.einsum("ji, Bi -> Bj", self.inv_weights, activation_grid)

    def spread_to_grid(self, activation_spread):
        if isinstance(activation_spread, torch.Tensor):

            return torch.einsum("ij, Bjc -> Bic", self.weight_torch.to(activation_spread.device),
                                activation_spread).reshape(activation_spread.shape[0], 24, 24, 24, activation_spread.shape[-1])
        if len(activation_spread.shape) == 1:
            return np.einsum("ij, j -> i", self.weights, activation_spread)
        if len(activation_spread.shape) == 2:
            return np.einsum("ij, Bj -> Bi", self.weights, activation_spread)

    def grid_to_spread(self, activation_grid):
        return activation_grid[..., self.indices]

    def spreadlf_to_grid(self, lf_spread):
        if len(lf_spread.shape) == 2:
            # We have fixed orientation
            return np.einsum("ij, Mj -> Mi", self.weights, lf_spread) * self.mask_grid[None]
        elif len(lf_spread.shape) == 3:
            # We have flexible orientation
            return np.einsum("ij, Mjc -> Mic", self.weights, lf_spread) * self.mask_grid[None]

    def get_pseudo_inv(self, lf):
        # We assume that the lf is only there for voxels with weight!
        return mkfilt_eloreta_v2(lf)


def get_coords_grid():
    x,y,z = (-0.06956156127119535, -0.0728710381912136, 0.0022035735789326426)
    # I should pixel align, the grid should start from the minimal value in the bottom left
    x_coords = np.arange(x, x + 24 * 0.007, 0.007)
    y_coords = np.arange(y, y + 24 * 0.007, 0.007)
    z_coords = np.arange(z, z + 24 * 0.007, 0.007)
    lx, ly, lz = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
    coords_grid = np.stack([lx, ly, lz], axis=3)
    # coords_grid
    return coords_grid.reshape(-1, 3)



class EsinetDataset(Dataset):
    """EEG dataset."""

    def __init__(self, root_dir="data", mode="train", oneD=False, sampling="ico4", normalize=False):
        self.sample_time = False
        if sampling == "ico3":
            self.data = {}
            tmp_data = np.load(root_dir + f"/Esinetico3Large{mode}.npz", allow_pickle=True)
            #(N,C,T) = tmp_data["measure"].shape
            # Shape is NSamples x Channels x Time, we reshape to Nsamples, Time, Channels
            self.data["measure"] = tmp_data["measure"].astype(np.float32).transpose(0, 2, 1)#.reshape(N*T, -1)
            self.data["sources"] = tmp_data["sources"].astype(np.float32).transpose(0, 2, 1)#.reshape(N*T, -1)
            if normalize:
                # TODO Axis should have been 2, averaging over channel !!!
                self.data["measure"] = ((self.data["measure"] - np.mean(self.data["measure"], axis=2, keepdims=True))
                                        / np.std(self.data["measure"], axis=2, keepdims=True))
                self.data["sources"] = self.data["sources"] / np.max(np.abs(self.data["sources"]), axis=2, keepdims=True)
                #self.data["measure"] = (self.data["measure"] - np.mean(self.data["measure"], axis=0, keepdims=True)) / np.std(self.data["measure"], axis=0, keepdims=True)
                #self.data["sources"] = self.data["sources"] / np.max(np.abs(self.data["sources"]), axis=0, keepdims=True)
            self.sample_time = True
        elif oneD:
            self.data = {}
            tmp_data = np.load(root_dir + f"/{mode}esinetdata.npy", allow_pickle=True)[()]
            self.data["measure"] = tmp_data["measure"].astype(np.float32)
            self.data["sources"] = tmp_data["sources"].astype(np.float32)
        else:
            #We try to preprocess the data to grid in order to have a speed up
            try:
                self.data = np.load(root_dir + f"/{mode}esinetdata_processed.npy", allow_pickle=True)[()]
            except:
                # Need to process data!
                tmp_data = np.load(root_dir + f"/{mode}esinetdata.npy", allow_pickle=True)[()]
                coords_grid = get_coords_grid()
                # TODO make sim work live?!
                sim, fwd = get_esinet_simulation()
                fwd_coords = fwd["source_rr"]
                TC = Transfer_Class(coords_grid, fwd_coords)

                tmp_grid_sources = np.zeros((tmp_data["sources"].shape[0], 1, 24, 24, 24))
                for s_idx in range(tmp_data["sources"].shape[0]):

                    tmp_grid_sources[s_idx] = TC.spread_to_grid(tmp_data["sources"][s_idx]).reshape(1,24,24,24)
                self.data = {}

                self.data["measure"] = tmp_data["measure"].astype(np.float32)
                self.data["sources"] = tmp_grid_sources.astype(np.float32)
                np.save(root_dir + f"/{mode}esinetdata_processed.npy", self.data)

        self.N = self.data["sources"].shape[0]

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        #sources = self.TC.spread_to_grid(self.data["sources"][idx]).reshape(1,24,24,24)
        if self.sample_time:
            time_step = np.random.randint(0, self.data["measure"].shape[1])
            sources = self.data["sources"][idx, time_step]
            measure = self.data["measure"][idx, time_step]
        else:
            sources = self.data["sources"][idx]
            measure = self.data["measure"][idx]
        sample = {
            "sensors": measure, #.astype(np.float32),
            "sources": sources[..., None], #.astype(np.float32),
            "Nsources": -1,
            "forward_index": 0
        }
        if "source_centers" in self.data.keys():
            sample["source_centers"] = self.data["source_centers"][idx]


        return sample


#get_esinet_data()