import numpy as np
import scipy
import torch
from scipy import signal
import mne
import os
from utils.bsi_zoo import gamma_map
from copy import deepcopy
def compute_sigma_displacement_matrix(PL):
    n_channel = np.shape(PL)[3]
    assert np.shape(PL)[4]==n_channel, "Somehow different number of channels of forward and inverse"
    sigma_matrix = np.zeros((*PL.shape[:3], n_channel,6)) # we have 3d coordinates and 3d variance
    indices = np.where(PL.sum(dim=(3,4,5,6,7))!=0)

    for i, j, k in zip(*indices):
        sigmas = []
        for l in range(n_channel):
            out = fit_gaussian_2(PL[i, j, k, l, l])
            # Calculate displacement
            out[:3] = out[:3] - np.asarray([i, j, k])
            sigmas.append(out)
        sigma_matrix[i, j, k, :, :] = np.asarray(sigmas)
    return sigma_matrix


def fit_gaussian_2(x):
    grid = torch.clamp(x, min=0).numpy()  # We remove negative values for simplicity
    # grid = x[0,0]
    grid = grid / grid.sum()  # We normalize to sum 1
    coords = np.indices(grid.shape)
    coords_flat = coords.reshape(3, -1)
    grid_flat = grid.flatten()
    mean = (coords_flat * grid_flat[None]).sum(1)
    var = ((coords_flat - mean[:, None]) ** 2 * grid_flat[None]).sum(1) ** 0.5
    #We need to revert mean and sigma as we have indices i,j,k,l,  l,k,j,i for matrix PL!
    return np.concatenate([mean[::-1], var[::-1]])


def reshape_fortran(x, shape):
    # TODO test
    if len(x.shape) > 1:
        x = x.permute(0, *reversed(range(1, len(x.shape))))
    return x.reshape(shape[0], *reversed(shape[1:])).permute(0, *reversed(range(1, len(shape))))


def gkern3d(kernlen=7, std=3):
    """Returns a 2D Gaussian kernel array."""
    gkern1d = signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
    gkern3d = np.outer(gkern1d, np.outer(gkern1d, gkern1d)).reshape((kernlen, kernlen, kernlen))
    gkern3d /= gkern3d.sum()
    return gkern3d


def get_3dkernel_matrix(img_shape, kernel, idx_to_calc=None):
    ksize = len(kernel)
    # Construct the sparse matrix A
    assert len(img_shape) == 3, "We assume 3D image"
    W, H, D = img_shape
    indices = []
    values = []
    if idx_to_calc is None:
        idx_to_calc = range(W * H * D)
    else:
        idx_to_calc = f_to_c_index(idx_to_calc, img_shape)
    for i in idx_to_calc:
        xi, yi, zi = np.unravel_index(i, img_shape)
        for j in range(max(0, xi - ksize // 2), min(W, xi + ksize // 2 + 1)):
            for k in range(max(0, yi - ksize // 2), min(H, yi + ksize // 2 + 1)):
                for l in range(max(0, zi - ksize // 2), min(D, zi + ksize // 2 + 1)):
                    jkl = np.ravel_multi_index((j, k, l), img_shape)
                    indices.append((i, jkl))
                    values.append(kernel[j - xi + ksize // 2, k - yi + ksize // 2, l - zi + ksize // 2])
    A = torch.sparse_coo_tensor(
        indices=torch.tensor(indices).t(),
        values=torch.tensor(values),
        size=(img_shape[0] * img_shape[1] * img_shape[2], img_shape[0] * img_shape[1] * img_shape[2]),
        dtype=torch.float32,
    )
    A = A.to_dense() / (A.to_dense().sum(0, keepdims=True) + 1e-10)
    return A


def f_to_c_index(indices, img_shape):
    long_idx = np.unravel_index(indices, img_shape, order="F")
    return np.ravel_multi_index(long_idx, img_shape)


def convert_3dkernel_to_flat(A_3d, img_shape, forward_model):
    A_3d_pre = A_3d[:, None].repeat(1, 3, 1).reshape(-1, 3, *img_shape)
    first_step = forward_model.volume_to_vector(A_3d_pre).flatten(1)
    sec_pre = first_step.T[:, None].repeat(1, 3, 1).reshape(-1, 3, *img_shape)
    sec_step = forward_model.volume_to_vector(sec_pre).flatten(1).T

    sec_step = sec_step / sec_step.sum(0, keepdims=True) * 3.0
    return sec_step


def mkfilt_eloreta_v2(L, regu=0.05):
    """

    % R.D. Pascual-Marqui: Discrete, 3D distributed, linear imaging methods of electric neuronal activity. Part 1: exact, zero
    % error localization. arXiv:0710.3341 [math-ph], 2007-October-17, http://arxiv.org/pdf/0710.3341
    """
    nchan, ng, ndum = L.shape
    LL = np.zeros((nchan, ndum, ng))
    for i in range(ndum):
        LL[:, i, :] = L[:, :, i]
    LL = np.reshape(LL, (nchan, ndum * ng), order='F')

    u0 = np.eye(nchan)
    W = np.reshape(np.tile(np.eye(ndum), (1, ng)), (ndum, ndum, ng), order='F')
    Winv = np.zeros((ndum, ndum, ng))
    winvkt = np.zeros((ng * ndum, nchan))
    kont = 0
    kk = 0
    while kont == 0:
        kk += 1
        for i in range(ng):
            Winv[:, :, i] = np.linalg.inv(W[:, :, i] + np.trace(W[:, :, i]) / (ndum * 1e6))
        for i in range(ng):
            winvkt[ndum * i:ndum * (i + 1), :] = np.dot(Winv[:, :, i], LL[:, ndum * i:ndum * (i + 1)].conj().T)
        kwinvkt = np.dot(LL, winvkt)
        alpha = regu * np.trace(kwinvkt) / nchan
        M = np.linalg.inv(kwinvkt + alpha * u0)
        ux, sx, vx = np.linalg.svd(kwinvkt)
        for i in range(ng):
            Lloc = L[:, i, :]
            Wold = np.copy(W)
            W[:, :, i] = np.real(scipy.linalg.sqrtm(np.dot(Lloc.conj().T, np.dot(M, Lloc))))
        reldef = np.linalg.norm(W.flatten() - Wold.flatten()) / np.linalg.norm(Wold.flatten())
        if kk > 20 or reldef < 1e-6:
            kont = 1

    ktm = np.dot(LL.conj().T, M)
    A = np.zeros((nchan, ng, ndum))
    for i in range(ng):
        A[:, i, :] = np.dot(Winv[:, :, i], ktm[ndum * i:ndum * (i + 1), :]).conj().T

    return A


def get_mne_src_fwd_inv(mode="train", eeg_data=None):

    fs_dir = mne.datasets.fetch_fsaverage(verbose=False)
    subjects_dir = os.path.dirname(fs_dir)
    subject = 'fsaverage'
    trans = os.path.join(fs_dir, 'bem', 'fsaverage-trans.fif')
    src = os.path.join(fs_dir, 'bem', 'fsaverage-ico-5-src.fif')
    # bem = f"fsaverage-5120-5120-5120-bem-sol_val.fif"
    # Create our own info object, see e.g.:
    info = get_info()

    vol_src = mne.setup_volume_source_space(subject, pos=7,
                                            subjects_dir=subjects_dir, verbose=False)

    # TODO in the future we can vary the conductivity
    if "val" in mode:
        conductivity = [0.332, 0.0113, 0.332]
    else:
        conductivity = [0.3, 0.006, 0.3]
    # TODO Maybe save and load bem model instead of creating it every time
    try:
        bem = mne.read_bem_solution(f"data/BEMfsaverage{mode}.fif")
    except FileNotFoundError:
        bem = mne.make_bem_solution(mne.make_bem_model("fsaverage", conductivity=conductivity, verbose=False), verbose=False)
        mne.write_bem_solution(f"data/BEMfsaverage{mode}.fif", bem, verbose=False)

    try:
        fwd = mne.read_forward_solution(f"data/fsaverageforward{mode}-fwd.fif")
    except:
        fwd = mne.make_forward_solution(info, trans=trans, src=vol_src,
                                        bem=bem, eeg=True, meg=False, mindist=5.0, n_jobs=1, verbose=False)
        mne.write_forward_solution(f"data/fsaverageforward{mode}-fwd.fif", fwd)

    if eeg_data is None:
        noise_cov = mne.make_ad_hoc_cov(info, verbose=0)
    else:

        if len(eeg_data.shape) == 4:
            print(f"We have an additional time dimension: {eeg_data.shape}")
            eeg_data = eeg_data.transpose(0,3,2,1)[:,0]

        eeg_tmp = mne.EpochsArray(eeg_data, info, verbose=0)
        eeg_tmp.set_eeg_reference(projection=True)
        eeg_tmp.apply_proj()
        noise_cov = mne.compute_covariance(
            eeg_tmp, tmax=0.0, method=["shrunk", "empirical"], rank=None, verbose=0
        )

    inv = mne.minimum_norm.make_inverse_operator(info, fwd, noise_cov, loose=1, fixed=False, depth=0.8, verbose=False)

    return {"src": vol_src, "fwd": fwd, "inv": inv, "info": info, "cov": noise_cov}

def get_info(kind='easycap-M10', sfreq=1000):
    ''' Create some generic mne.Info object.
    Parameters
    ----------
    kind : str
        kind, for examples see:
            https://mne.tools/stable/generated/mne.channels.make_standard_montage.html#mne.channels.make_standard_montage

    Return
    ------
    info : mne.Info
        The mne.Info object
    '''
    montage = mne.channels.make_standard_montage(kind)
    info = mne.create_info(montage.ch_names, sfreq, ch_types=['eeg'] * len(montage.ch_names), verbose=0)
    info.set_montage(kind)
    return info

def save_gamma_map(s, L, noise_var, n_channel, Nsources,alpha=0.2):
    #print(s.shape)
    #s = deepcopy(s)
    #noise_var = deepcopy(noise_var)
    #L = deepcopy(L)
    return gamma_map(L=L, y=s, n_orient=n_channel, alpha=alpha, cov=noise_var)
    try:
        tmp_out = gamma_map(L=L, y=s, n_orient=n_channel, alpha=0.2, cov=noise_var)
    except np.linalg.LinAlgError:
        #print(f"Error in gamma map, using pseudoinv instead, there are {np.isnan(sens).any()} NANS in Sensors")
        tmp_out = np.linalg.pinv(L) @ s
        tmp_out = tmp_out.reshape(Nsources, n_channel)
    return tmp_out


def total_variation_3d(x):
    """
    Compute the total variation loss for a 4D tensor.

    Args:
        x (torch.Tensor): A 4D tensor of shape (B, D, H, W).

    Returns:
        torch.Tensor: The total variation loss.
    """
    # Compute differences in the x, y, and z directions
    diff_x = x[...,1:, :, :] - x[...,:-1, :, :]
    diff_y = x[...,:, 1:, :] - x[...,:, :-1, :]
    diff_z = x[...,:, :, 1:] - x[...,:, :, :-1]

    # Compute the TV norm
    tv_norm = torch.sum(
        torch.sqrt(diff_x ** 2 + diff_y ** 2 + diff_z ** 2 + 1e-8))  # add small epsilon for numerical stability

    return tv_norm



def champ_vec(y, f, sigu, nem, nd):
    eps1 = 1e-8
    nk, nvd = f.shape
    nv = nvd // nd
    nt = y.shape[1]

    cyy = y @ y.T / nt

    # Initialize voxel variances
    f2 = np.sum(f ** 2, axis=0)
    invf2 = np.zeros(nvd)
    ff = f2 > 0
    invf2[ff] = 1. / f2[ff]
    f = f.astype(float)
    w = np.diag(invf2) @ f.T
    inu0 = np.mean(np.mean((w @ y) ** 2))
    vvec = inu0 * np.ones(nvd)
    like = np.zeros(nem)
    for iem in range(nem):

        vmat = np.diag(vvec)
        c = f @ vmat @ f.T + sigu
        d, p = scipy.linalg.eig(c)
        d = np.maximum(np.real(d), 0)
        invd = np.zeros(nk)
        ff = d >= eps1
        invd[ff] = 1. / d[ff]
        invc = p @ np.diag(invd) @ p.T

        like[iem] = -0.5 * (np.sum(np.log(np.maximum(d, eps1))) + nk * np.log(2 * np.pi)) - 0.5 * np.sum(invc * cyy)

        fc = f.T @ invc
        w = vmat @ fc
        x = w @ y
        x2 = np.mean(x ** 2, axis=1)
        z = np.sum(fc * f.T, axis=1)

        vvec = np.zeros_like(x2)
        ff = z > 0
        vvec[ff] = np.sqrt(np.maximum(x2[ff] / z[ff], 0))


    return x
