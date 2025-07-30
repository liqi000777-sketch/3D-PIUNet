import torch
import geomloss
import numpy as np
import warnings
import torchmetrics

from sklearn.metrics import auc, roc_curve
from scipy.spatial.distance import cdist
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import ot
def compute_metrics(gt, pred, forward_model=None, forward_index = None, measurements = None, compute_emd = False):

    result = {}
    if len(gt.shape) >= 4 and forward_model is not None:
        result.update({
            "OptimalTransport": ot_loss(gt, pred).detach().cpu(),
            "ActivationDistance": maximal_activation_distance(gt, pred).detach().cpu(),
            "SSIM": torchmetrics.functional.structural_similarity_index_measure(pred, gt,
                                                                                reduction="none").detach().cpu(),
        })

    # This way, we filter out all the border voxels and only consider real sources
    # gt = forward_model.volume_to_vector(gt)
    # pred = forward_model.volume_to_vector(pred)

    result.update({
        "MAE": mean_absolut_error(gt, pred).detach().cpu(),
        "MSE": mean_squared_error(gt, pred).detach().cpu(),
        "NormalizedMAE": normalized_absolut_error(gt, pred).detach().cpu(),
        "NormalizedMSE": normalized_mean_squared_error(gt, pred).detach().cpu(),
        "VNMAE":variability_normalized_absolut_error(gt,pred).detach().cpu(),
        "VNMSE": variability_normalized_mean_squared_error(gt, pred).detach().cpu(),
        "UnitMSE": unit_normalized_mean_squared_error(gt, pred).detach().cpu(),
        "AngularError": calculate_angular_error(gt, pred).detach().cpu(),
        "WeightedAngularError": calculate_angular_error(gt, pred, mode="weighted").detach().cpu(),
        #"PSNR": torchmetrics.functional.peak_signal_noise_ratio(pred, gt, reduction=None).flatten(1).mean(1).detach().cpu(),  # We need to account for reduction

        })
    if forward_model is not None:
        result.update({
            "CleanMeasurementError": mean_squared_error(forward_model.forward_specific(pred, forward_index),forward_model.forward_specific(gt, forward_index)),
        "CleanNormalizedMeasurementError": normalized_mean_squared_error(
            forward_model.forward_specific(pred, forward_index),forward_model.forward_specific(gt, forward_index))
        })
        if measurements is not None:
            result.update({
                "MeasurementError": mean_squared_error(forward_model.forward_specific(pred, forward_index),
                                                       measurements),
                "NormalizedMeasurementError": normalized_mean_squared_error(
                    forward_model.forward_specific(pred, forward_index), measurements)
            })

    if forward_model is not None and compute_emd: #Computing sinkhorn and EMD is to slow :/
        emd, normalized_emd = compute_emd_and_sinkhorn(gt, pred, forward_model)
        result.update({
            "EMD": torch.Tensor(emd),
            "NormalizedEMD": torch.Tensor(normalized_emd)
        })
        #esinet_metrics = get_various_metrics(gt, pred, forward_model,forward_index=forward_index)
        #result.update(esinet_metrics)

    return result

def compute_emd_and_sinkhorn(gt, pred, forward_model):
    # We can compute the optimal transport and earth mover distance
    emd = np.zeros(gt.shape[0])
    sinkhorn = np.zeros(gt.shape[0])
    #flat_gt = forward_model.volume_to_vector(gt).reshape(gt.shape[0],-1,forward_model.n_channel).norm(dim=2).detach().cpu().numpy()
    #flat_pred = forward_model.volume_to_vector(pred).reshape(gt.shape[0],-1,forward_model.n_channel).norm(dim=2).detach().cpu().numpy()
    flat_gt = np.float64(gt.reshape(gt.shape[0], -1, forward_model.n_channel).norm(dim=2).detach().cpu().numpy())
    flat_pred = np.float64(pred.reshape(gt.shape[0], -1, forward_model.n_channel).norm(dim=2).detach().cpu().numpy())
    dis_matrix = forward_model.dis_matrix.cpu().numpy()
    a = flat_pred / np.sum(flat_pred, axis=1, keepdims=True)
    b = flat_gt / np.sum(flat_gt, axis=1, keepdims=True)

    normalized_pred = np.ones_like(flat_pred) / flat_pred.shape[1]
    with ThreadPoolExecutor() as executor:
        # Map the function over the range of indices, might speed up?
        results = list(
            executor.map(lambda i: calculate_ot_distances(i, a, b, dis_matrix), range(gt.shape[0])))

        normalization_factor = list(
            executor.map(lambda i: calculate_ot_distances(i, normalized_pred, b, dis_matrix), range(gt.shape[0])))
    #sinkhorn = np.asarray(results)
    emd = np.asarray(results)
    emd_normalized = emd / np.asarray(normalization_factor)
    return emd, emd_normalized

    for i in range(gt.shape[0]):
        # No batched operation, we have to loop
        # TODO emd to slow! sinkhorn also slow!
        emd[i] = ot.emd2(a, b, forward_model.dis_matrix)
        #Maybe sinkhorn is even slower than emd?
        # sinkhorn[i] = ot.sinkhorn2(a[i], b[i], dis_matrix, 0.1, method='sinkhorn_stabilized')
    return emd, sinkhorn


def calculate_ot_distances(i, a, b, dis_matrix):
    emd_i = ot.emd2(a[i], b[i], dis_matrix)
    return emd_i
    #sinkhorn_i = ot.sinkhorn2(a[i], b[i], dis_matrix, 0.1, method='sinkhorn_stabilized')
    return sinkhorn_i

def mean_absolut_error(gt, pred):
    gt = gt.flatten(1)
    pred = pred.flatten(1)
    return torch.mean(torch.abs(gt - pred), dim=1)


def mean_squared_error(gt, pred):
    gt = gt.flatten(1)
    pred = pred.flatten(1)
    return torch.mean((gt - pred) ** 2, dim=1)


def normalized_absolut_error(gt, pred):
    gt = gt.flatten(1)
    pred = pred.flatten(1)
    return torch.mean(torch.abs(gt - pred), dim=1) / torch.mean(torch.abs(gt), dim=1)


def normalized_mean_squared_error(gt, pred):
    gt = gt.flatten(1)
    pred = pred.flatten(1)
    return torch.mean((gt - pred) ** 2, dim=1) / torch.mean(gt**2, dim=1)

def variability_normalized_absolut_error(gt, pred):
    gt = gt.flatten(1)
    pred = pred.flatten(1)
    return torch.mean(torch.abs(gt - pred), dim=1) / torch.mean(torch.abs(gt-torch.mean(gt, dim=1, keepdim=True)), dim=1)


def variability_normalized_mean_squared_error(gt, pred):
    gt = gt.flatten(1)
    pred = pred.flatten(1)
    return torch.mean((gt - pred) ** 2, dim=1) / torch.mean((gt-torch.mean(gt, dim=1, keepdim=True))**2, dim=1)

def unit_normalized_mean_squared_error(gt, pred):
    gt = gt.flatten(1).abs()
    pred = pred.flatten(1).abs()
    gt /= torch.max(gt, dim=1, keepdim=True)[0]
    pred /= torch.max(pred, dim=1, keepdim=True)[0]
    return torch.mean((gt - pred) ** 2, dim=1)

def ot_loss(gt, pred):
    try:
        loss = geomloss.SamplesLoss()
        return loss(gt.flatten(2), pred.flatten(2))
    except Exception as e:
        warnings.warn(f"Could not compute OT loss {e}")
        return torch.zeros(gt.shape[0], device=gt.device)


def maximal_activation_distance(gt, pred):
    gt = torch.linalg.vector_norm(gt, dim=2)
    pred = torch.linalg.vector_norm(pred, dim=2)
    return torch.linalg.vector_norm((get_max_activation_coordinates(gt) - get_max_activation_coordinates(pred)), dim=1)


def get_max_activation_coordinates(x):
    """Input a batch of 3D voxels (N,W,H,D)
    Outputs a Batch of 3D coordinates (N,3)"""
    indices = torch.argmax(x.flatten(1), dim=1).cpu().numpy()
    return torch.tensor(np.asarray(np.unravel_index(indices, x.shape[1:])).T, dtype=torch.float, device=x.device)






def get_various_metrics(gt, pred, forward_model, forward_index):
    measure_true = forward_model.forward_specific(gt,forward_index).detach().cpu().numpy()
    measure_pred = forward_model.forward_specific(pred, forward_index).detach().cpu().numpy()
    gt = forward_model.volume_to_vector(gt).norm(dim=2).detach().cpu().numpy()
    pred = forward_model.volume_to_vector(pred).norm(dim=2).detach().cpu().numpy()

    if not hasattr(forward_model, "argsorted_distance_matrix"):
        forward_model.argsorted_distance_matrix = forward_model.dis_matrix.cpu().numpy().argsort(axis=1)
    results = {"measurement_residual_variance": [],
               "mean_localization_error": [],
               "ghost_sources": [],
               "found_sources": [],
               "auc": [],
               }
    for i in range(measure_pred.shape[0]):
        if pred[i].sum() == 0 or gt[i].sum() == 0:
            warn_msg = f"We have a zero in pred {pred[i].sum()} or GT {gt[i].sum()}"
            warnings.warn(warn_msg, RuntimeWarning)
            continue

        results["measurement_residual_variance"].append(eval_residual_variance(measure_true[i], measure_pred[i]))

        mean_localization_error, ghost_sources, found_sources = eval_mean_localization_error(gt[i], pred[i], forward_model.scs_loc, argsorted_distance_matrix=forward_model.argsorted_distance_matrix)
        results["mean_localization_error"].append(mean_localization_error)
        results["ghost_sources"].append(ghost_sources)
        results["found_sources"].append(found_sources)
        results["auc"].append(eval_auc(gt[i], pred[i], forward_model.scs_loc))

    results = {key: torch.Tensor(results[key]) for key in results.keys()}
    return results


############## Code from esinet LukeTheHecker
############## https://github.com/LukeTheHecker/esinet/blob/main/esinet/evaluate/evaluate.py



def get_maxima_mask(y, pos, k_neighbors=5, threshold=0.1, min_dist=0.03,
                    argsorted_distance_matrix=None):
    ''' Returns the mask containing the source maxima (binary).
    Parameters
    ----------
    y : numpy.ndarray
        The source
    pos : numpy.ndarray
        The dipole position matrix
    k_neighbors : int
        The number of neighbors to incorporate for finding maximum
    threshold : float
        Proportion between 0 and 1. Defined the minimum value for a maximum to
        be of significance. 0.1 -> 10% of the absolute maximum
    '''
    if argsorted_distance_matrix is None:
        argsorted_distance_matrix = np.argsort(cdist(pos, pos), axis=1)

    y = np.abs(y)
    threshold = threshold * np.max(y)
    # find maxima that surpass the threshold:
    close_idc = argsorted_distance_matrix[:, 1:k_neighbors + 1]
    mask = ((y >= np.max(y[close_idc], axis=1)) & (y > threshold)).astype(int)
    # filter maxima
    maxima = np.where(mask == 1)[0]
    distance_matrix_maxima = cdist(pos[maxima], pos[maxima])
    for i, _ in enumerate(maxima):
        distances_maxima = distance_matrix_maxima[i]
        close_maxima = maxima[np.where(distances_maxima < min_dist)[0]]
        # If there is a larger maximum in the close vicinity->delete maximum
        if np.max(y[close_maxima]) > y[maxima[i]]:
            mask[maxima[i]] = 0

    return mask

def get_maxima_pos(mask, pos):
    ''' Returns the positions of the maxima within mask.
    Parameters
    ----------
    mask : numpy.ndarray
        The source mask
    pos : numpy.ndarray
        The dipole position matrix
    '''
    return pos[np.where(mask == 1)[0]]


def eval_residual_variance(M_true, M_est):
    ''' Calculate the Residual Variance (1- goodness of fit) between the
    estimated EEG and the original EEG.

    Parameters
    ----------
    M_true : numpy.ndarray
        The true EEG data (as recorded). May be a single time point or
        spatio-temporal.
    M_est : numpy.ndarray
        The estimated EEG data (projected from the estimated source). May be a
        single time point or spatio-temporal.
    '''
    return 100 * np.sum((M_true - M_est) ** 2) / np.sum(M_true ** 2)


def eval_mean_localization_error(y_true, y_est, pos, k_neighbors=5,
                                 min_dist=0.03, threshold=0.1, ghost_thresh=0.04, argsorted_distance_matrix=None):
    ''' Calculate the mean localization error for an arbitrary number of
    sources.

    Parameters
    ----------
    y_true : numpy.ndarray
        The true source vector (1D)
    y_est : numpy.ndarray
        The estimated source vector (1D)
    pos : numpy.ndarray
        The dipole position matrix
    k_neighbors : int
        The number of neighbors to incorporate for finding maximum
    threshold : float
        Proportion between 0 and 1. Defined the minimum value for a maximum to
        be of significance. 0.1 -> 10% of the absolute maximum
    min_dist : float/int
        The minimum viable distance in mm between maxima. The higher this
        value, the more maxima will be filtered out.
    ghost_thresh : float/int
        The threshold distance between a true and a predicted source to not
        belong together anymore. Predicted sources that have no true source
        within the vicinity defined be ghost_thresh will be labeled
        ghost_source.

    Return
    ------
    mean_localization_error : float
        The mean localization error between all sources in y_true and the
        closest matches in y_est.
    '''
    y_true = deepcopy(y_true)
    y_est = deepcopy(y_est)

    maxima_true = get_maxima_pos(
        get_maxima_mask(y_true, pos, k_neighbors=k_neighbors,
                        threshold=threshold, min_dist=min_dist,
                        argsorted_distance_matrix=argsorted_distance_matrix), pos)
    maxima_est = get_maxima_pos(
        get_maxima_mask(y_est, pos, k_neighbors=k_neighbors,
                        threshold=threshold, min_dist=min_dist,
                        argsorted_distance_matrix=argsorted_distance_matrix), pos)

    # Distance matrix between every true and estimated maximum
    distance_matrix = cdist(maxima_true, maxima_est)
    # For each true source find the closest predicted source:
    closest_matches = distance_matrix.min(axis=1)

    # Filter ghost sources
    ghost_sources = closest_matches[closest_matches >= ghost_thresh]
    n_ghost_sources = len(ghost_sources)


    # Filter ghost sources
    found_sources = closest_matches[closest_matches < ghost_thresh]
    n_found_sources = len(found_sources)

    # No source left -> return nan
    if len(found_sources) == 0:
        mean_localization_error = np.nan
    else:
        mean_localization_error = np.mean(found_sources)

    return mean_localization_error, n_ghost_sources, n_found_sources





def eval_nmse(y_true, y_est):
    '''Returns the normalized mean squared error between predicted and true
    source.'''

    y_true_normed = y_true / np.max(np.abs(y_true))
    y_est_normed = y_est / np.max(np.abs(y_est))
    return np.mean((y_true_normed - y_est_normed) ** 2)


def eval_auc(y_true, y_est, pos, n_redraw=25, epsilon=0.25,
             plot_me=False):
    ''' Returns the area under the curve metric between true and predicted
    source.

    Parameters
    ----------
    y_true : numpy.ndarray
        True source vector
    y_est : numpy.ndarray
        Estimated source vector
    pos : numpy.ndarray
        Dipole positions (points x dims)
    n_redraw : int
        Defines how often the negative samples are redrawn.
    epsilon : float
        Defines threshold on which sources are considered
        active.
    Return
    ------
    auc_close : float
        Area under the curve for dipoles close to source.
    auc_far : float
        Area under the curve for dipoles far from source.
    '''

    if y_est.sum() == 0 or y_true.sum() == 0:
        return np.nan, np.nan
    y_true = deepcopy(y_true)
    y_est = deepcopy(y_est)
    # Absolute values
    y_true = np.abs(y_true)
    y_est = np.abs(y_est)

    # Normalize values
    y_true /= np.max(y_true)
    y_est /= np.max(y_est)

    auc_close = np.zeros((n_redraw))
    auc_far = np.zeros((n_redraw))

    # t_prep = time.time()
    # print(f'\tprep took {1000*(t_prep-t_start):.1f} ms')

    source_mask = (y_true > epsilon).astype(int)

    numberOfActiveSources = int(np.sum(source_mask))
    # print('numberOfActiveSources: ', numberOfActiveSources)
    numberOfDipoles = pos.shape[0]
    # Draw from the 20% of closest dipoles to sources (~100)
    closeSplit = int(round(numberOfDipoles / 5))
    # Draw from the 50% of furthest dipoles to sources
    farSplit = int(round(numberOfDipoles / 2))
    # t_prep = time.time()
    # print(f'\tprep took {1000*(t_prep-t_start):.1f} ms')

    distSortedIndices = find_indices_close_to_source(source_mask, pos)

    # t_prep2 = time.time()
    # print(f'\tprep2 took {1000*(t_prep2-t_prep):.1f} ms')

    sourceIndices = np.where(source_mask == 1)[0]

    for n in range(n_redraw):
        selectedIndicesClose = np.concatenate(
            [sourceIndices, np.random.choice(distSortedIndices[:closeSplit], size=numberOfActiveSources)])
        selectedIndicesFar = np.concatenate(
            [sourceIndices, np.random.choice(distSortedIndices[-farSplit:], size=numberOfActiveSources)])
        # print(f'redraw {n}:\ny_true={y_true[selectedIndicesClose]}\y_est={y_est[selectedIndicesClose]}')
        fpr_close, tpr_close, _ = roc_curve(source_mask[selectedIndicesClose], y_est[selectedIndicesClose])

        fpr_far, tpr_far, _ = roc_curve(source_mask[selectedIndicesFar], y_est[selectedIndicesFar])

        auc_close[n] = auc(fpr_close, tpr_close)
        auc_far[n] = auc(fpr_far, tpr_far)

    auc_far = np.mean(auc_far)
    auc_close = np.mean(auc_close)

    if plot_me:
        import matplotlib.pyplot as plt
        print("plotting")
        plt.figure()
        plt.plot(fpr_close, tpr_close, label='ROC_close')
        plt.plot(fpr_far, tpr_far, label='ROC_far')
        # plt.xlim(1, )
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'AUC_close={auc_close:.2f}, AUC_far={auc_far:.2f}')
        plt.legend()
        plt.show()

    return auc_close, auc_far


def find_indices_close_to_source(source_mask, pos):
    ''' Finds the dipole indices that are closest to the active sources.

    Parameters
    -----------
    simSettings : dict
        retrieved from the simulate_source function
    pos : numpy.ndarray
        list of all dipole positions in XYZ coordinates

    Return
    -------
    ordered_indices : numpy.ndarray
        ordered list of dipoles that are near active
        sources in ascending order with respect to their distance to the next source.
    '''

    numberOfDipoles = pos.shape[0]

    sourceIndices = np.array([i[0] for i in np.argwhere(source_mask == 1)])

    min_distance_to_source = np.zeros((numberOfDipoles))

    # D = np.zeros((numberOfDipoles, len(sourceIndices)))
    # for i, idx in enumerate(sourceIndices):
    #     D[:, i] = np.sqrt(np.sum(((pos-pos[idx])**2), axis=1))
    # min_distance_to_source = np.min(D, axis=1)
    # min_distance_to_source[source_mask==1] = np.nan
    # numberOfNans = source_mask.sum()

    ###OLD
    numberOfNans = 0
    for i in range(numberOfDipoles):
        if source_mask[i] == 1:
            min_distance_to_source[i] = np.nan
            numberOfNans += 1
        elif source_mask[i] == 0:
            distances = np.sqrt(np.sum((pos[sourceIndices, :] - pos[i, :]) ** 2, axis=1))
            min_distance_to_source[i] = np.min(distances)
        else:
            print('source mask has invalid entries')
    # print('new: ', np.nanmean(min_distance_to_source), min_distance_to_source.shape)
    ###OLD

    ordered_indices = np.argsort(min_distance_to_source)

    return ordered_indices[:-numberOfNans]


def calculate_angular_error(gt, pred, mode="mask"):
    """
    Calculates the mean angular error between corresponding source vectors in
    ground truth and predicted source activations.

    Args:
      gt_sources: Torch array of shape (B, N, 3) representing ground truth source vectors.
      predicted_sources: Torch array of the same shape representing predicted source vectors.

    Returns:
      float: Cosine Similarity
    """
    gt_norm = torch.norm(gt, dim=2)
    cosine_similarity = torch.nn.functional.cosine_similarity(pred, gt, dim=2)
    if mode == "mask":
        # We consider only active regions
        mask = gt_norm > 1e-5
        cosine_similarity = cosine_similarity*mask
        mean_similarity = torch.sum(cosine_similarity, dim=1) / torch.sum(mask, dim=1)
    else:
        # We weight by percentage
        mask = gt_norm / gt_norm.sum(dim=1, keepdim=True)
        mean_similarity = torch.sum(cosine_similarity * mask, dim=1)

    return mean_similarity