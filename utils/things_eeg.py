import numpy as np
import mne
import os
from sklearn.utils import shuffle
from tqdm import tqdm
from sklearn.discriminant_analysis import _cov
from scipy.signal import find_peaks
import scipy
from conditioning.utils import mkfilt_eloreta_v2

def epoching(data_part, seed,ses=1):
    """This function first converts the EEG data to MNE raw format, and
    performs channel selection, epoching, baseline correction and frequency
    downsampling. Then, it sorts the EEG data of each session according to the
    image conditions.

    Parameters
    ----------
    data_part : str
        'test' or 'training' data partitions.
    seed : int
        Random seed.

    Returns
    -------
    epoched_data : list of float
        Epoched EEG data.
    img_conditions : list of int
        Unique image conditions of the epoched and sorted EEG data.
    ch_names : list of str
        EEG channel names.
    times : float
        EEG time points.


    Modifed from: https://github.com/gifale95/eeg_encoding/tree/main
    Gifford AT, Dwivedi K, Roig G, Cichy RM. 2022. A large and rich EEG dataset for modeling human visual object recognition.
    NeuroImage, 264:119754. DOI: https://doi.org/10.1016/j.neuroimage.2022.119754
    """



    ### Load the EEG data and convert it to MNE raw format ###
    eeg_dir = os.path.join('data/ThingsEEG2sub-10/', 'ses-' + format(ses, '02'), 'raw_eeg_' +
                           data_part + '.npy')
    eeg_data = np.load(eeg_dir, allow_pickle=True).item()
    ch_names = eeg_data['ch_names']
    sfreq = eeg_data['sfreq']
    ch_types = eeg_data['ch_types']
    eeg_data = eeg_data['raw_eeg_data']
    # Convert to MNE raw format
    info = mne.create_info(ch_names, sfreq, ch_types)
    raw = mne.io.RawArray(eeg_data, info)
    del eeg_data

    ### Get events, drop unused channels and reject target trials ###
    events = mne.find_events(raw, stim_channel='stim')
    # Reject the target trials (event 99999)
    idx_target = np.where(events[:, 2] == 99999)[0]
    events = np.delete(events, idx_target, 0)

    ### Epoching, baseline correction and resampling ###
    epochs = mne.Epochs(raw, events, tmin=-.2, tmax=.8, baseline=(None, 0),
                        preload=True)
    del raw
    # Resampling
    if sfreq < 1000:
        epochs.resample(sfreq)
    ch_names = epochs.info['ch_names']
    times = epochs.times

    ### Sort the data ###
    data = epochs.get_data()
    events = epochs.events[:, 2]
    img_cond = np.unique(events)
    del epochs
    # Select only a maximum number of EEG repetitions
    if data_part == 'test':
        max_rep = 20
    else:
        max_rep = 2
    # Sorted data matrix of shape:
    # Image conditions × EEG repetitions × EEG channels × EEG time points
    sorted_data = np.zeros((len(img_cond), max_rep, data.shape[1],
                            data.shape[2]))
    for i in range(len(img_cond)):
        # Find the indices of the selected image condition
        idx = np.where(events == img_cond[i])[0]
        # Randomly select only the max number of EEG repetitions
        idx = shuffle(idx, random_state=seed, n_samples=max_rep)
        sorted_data[i] = data[idx]
    del data
    sorted_data
    img_cond

    ### Output ###
    return sorted_data, img_cond, ch_names, times


def mvnn(epoched_test, mvnn_dim="time"):
    """Compute the covariance matrices of the EEG data (calculated for each
    time-point or epoch/repetitions of each image condition), and then average
    them across image conditions and data partitions. The inverse of the
    resulting averaged covariance matrix is used to whiten the EEG data
    (independently for each session).

    Parameters
    ----------
    args : Namespace
        Input arguments.
    epoched_test : list of floats
        Epoched test EEG data.
    epoched_train : list of floats
        Epoched training EEG data.

    Returns
    -------
    whitened_test : list of float
        Whitened test EEG data.
    whitened_train : list of float
        Whitened training EEG data.

    Modifed from : https://github.com/gifale95/eeg_encoding/tree/main
    Gifford AT, Dwivedi K, Roig G, Cichy RM. 2022. A large and rich EEG dataset for modeling human visual object recognition.
    NeuroImage, 264:119754. DOI: https://doi.org/10.1016/j.neuroimage.2022.119754
    """


    ### Loop across data collection sessions ###

    ### Compute the covariance matrices ###
    # Data partitions covariance matrix of shape:
    # Data partitions × EEG channels × EEG channels
    # Image conditions covariance matrix of shape:
    # Image conditions × EEG channels × EEG channels
    sigma_cond = np.empty((epoched_test.shape[0],
                           epoched_test.shape[2], epoched_test.shape[2]))
    for i in tqdm(range(epoched_test.shape[0])):
        cond_data = epoched_test[i]
        # Compute covariace matrices at each time point, and then
        # average across time points
        if mvnn_dim == "time":
            sigma_cond[i] = np.mean([_cov(cond_data[:, :, t],
                                          shrinkage='auto') for t in range(cond_data.shape[2])],
                                    axis=0)
        # Compute covariace matrices at each epoch (EEG repetition),
        # and then average across epochs/repetitions
        elif mvnn_dim == "epochs":
            sigma_cond[i] = np.mean([_cov(np.transpose(cond_data[e]),
                                          shrinkage='auto') for e in range(cond_data.shape[0])],
                                    axis=0)
    # Average the covariance matrices across image conditions
    sigma_tot = sigma_cond.mean(axis=0)
    # Compute the inverse of the covariance matrix
    sigma_inv = scipy.linalg.fractional_matrix_power(sigma_tot, -0.5)

    ### Whiten the data ###
    whitened_test = np.reshape((np.reshape(epoched_test, (-1,
                                                          epoched_test.shape[2], epoched_test.shape[3])).swapaxes(1, 2)
                                @ sigma_inv).swapaxes(1, 2), epoched_test.shape)
    ### Output ###
    return whitened_test


def get_things_eeg_info():
    # Specify the montage kind and sampling frequency
    kind = 'easycap-M1'
    sfreq = 1000

    ch_things_eeg = ['Fp1', 'F3', 'F7', 'FT9', 'FC5', 'FC1', 'C3', 'T7', 'TP9', 'CP5', 'CP1', 'Pz', 'P3', 'P7', 'O1',
                     'Oz', 'O2', 'P4', 'P8', 'TP10', 'CP6', 'CP2', 'Cz', 'C4', 'T8', 'FT10', 'FC6', 'FC2', 'F4', 'F8',
                     'Fp2', 'AF7', 'AF3', 'AFz', 'F1', 'F5', 'FT7', 'FC3', 'FCz', 'C1', 'C5', 'TP7', 'CP3', 'P1', 'P5',
                     'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'P6', 'P2', 'CPz', 'CP4', 'TP8', 'C6', 'C2', 'FC4', 'FT8', 'F6',
                     'F2', 'AF4', 'AF8']
    # Create the montage
    montage = mne.channels.make_standard_montage(kind)
    mapping = []
    for i in ch_things_eeg:
        mapping.append(montage.ch_names.index(i))
    m = montage.copy()
    m.ch_names = [m.ch_names[i] for i in mapping]
    m.dig = m.dig[:3] + [m.dig[i + 3] for i in mapping]

    # Create the info object
    info = mne.create_info(m.ch_names, sfreq, ch_types=['eeg'] * len(m.ch_names), verbose=0)
    # Set the montage to the info object
    info.set_montage(montage)
    return info

def generate_matching_mne_forward(filename="fsaverage_volume_thingseeg"):
    info = get_things_eeg_info()
    # Generate the forward model
    fs_dir = mne.datasets.fetch_fsaverage(verbose=False)
    subjects_dir = os.path.dirname(fs_dir)
    subject = 'fsaverage'
    trans = os.path.join(fs_dir, 'bem', 'fsaverage-trans.fif')

    vol_src = mne.setup_volume_source_space(subject, pos=7,
                                            subjects_dir=subjects_dir, verbose=False)


    try:
        bem_train = mne.read_bem_solution(f"data/BEMfsaverage_things_egg.fif")
    except FileNotFoundError:
        bem_train = mne.make_bem_solution(mne.make_bem_model("fsaverage", conductivity=[0.332, 0.0113, 0.332], verbose=False), verbose=False)
        mne.write_bem_solution(f"data/BEMfsaverage_things_egg.fif", bem_train, verbose=False)

    try:
        fwd_train = mne.read_forward_solution(f"data/fsaverageforward_things_egg-fwd.fif")
    except:
        fwd_train = mne.make_forward_solution(info, trans=trans, src=vol_src,
                                        bem=bem_train, eeg=True, meg=False, mindist=5.0, n_jobs=1, verbose=False)
        mne.write_forward_solution(f"data/fsaverageforward_things_egg-fwd.fif", fwd_train)


    if filename:
        sens_pos = np.array(list(info.get_montage()._get_ch_pos().values()))
        pinv = mkfilt_eloreta_v2(fwd_train["sol"]["data"].reshape(63, -1, 3))
        train_data = {"LeadField": fwd_train["sol"]["data"][:,:,None],
                      "PseudoInv": pinv,
                      "grid_loc": fwd_train["source_rr"],
                      "sens_loc": sens_pos,
                      "source_inds": fwd_train["src"][0]["vertno"],
                      "source_mask": fwd_train["src"][0]["inuse"].reshape(29,29,29)}
        np.save(f"data/{filename}.npy", train_data)

    return fwd_train

def prepare_data(mode="test", ses=1):
    if os.path.exists(f"data/ThingsEEG2sub-10/ses-0{ses}/whiten_data_{mode}.npz"):
        whiten_data, img_cond, ch_names, times, stimulus = np.load(f"data/ThingsEEG2sub-10/ses-0{ses}/whiten_data_{mode}.npz", allow_pickle=True).values()
    else:
        sorted_data, img_cond, ch_names, times = epoching(mode, 0, ses=ses)

        whiten_data = mvnn(sorted_data[:,:,:-1])
        stimulus = sorted_data[:,:,-1]
        np.savez(f"data/ThingsEEG2sub-10/ses-0{ses}/whiten_data_{mode}.npz", whiten_data, img_cond, ch_names, times, stimulus )


    return whiten_data, img_cond, ch_names, times, stimulus

def get_post_stimulus_data(mode="test", averaged=False, L=200, ses=1):
    if ses == 5:  # All sessions
        sets = []
        for i in range(1, 4):
            whiten_data, img_cond, ch_names, times, stimulus = prepare_data(mode, ses=i)
            sets.append(whiten_data[:L, :, :, :600])
        post_simulus_subset = np.concatenate(sets, axis=1)
    else:
        whiten_data, img_cond, ch_names, times, stimulus = prepare_data(mode, ses=ses)
        post_simulus_subset = whiten_data[:L, :, :, :600]
    if averaged:
        return post_simulus_subset.mean(axis=1).reshape(-1, 63, 600).transpose(0, 2, 1)
    return post_simulus_subset.reshape(-1, 63, 600).transpose(0, 2, 1)

def find_first_stim_data(raw_eeg_data):
    stimulus = raw_eeg_data[-1]

    stim = np.where(np.abs(stimulus - 200) < 200)[0]
    fil_stim = [b for a, b in zip(stim[:-1], stim[1:]) if b - a > 450]
    first_stim_data = [raw_eeg_data[:63][:, x:x + 400] for x in fil_stim]
    first_stim_data = np.array(first_stim_data)
    return first_stim_data

def find_peaks(data):
    # Channel 15 ist the most active after showing the image
    avg_channel = np.mean(data[:, :, :63], axis=(0, 1))
    return find_peaks(np.abs(np.mean(avg_channel, axis=0)), distance=40)[0]


