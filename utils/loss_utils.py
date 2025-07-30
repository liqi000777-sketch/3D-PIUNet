import torch.nn as nn
import torch
import torchvision
def cdist(x, y):
    '''
    Input: x is a Nxd Tensor
           y is a Mxd Tensor
    Output: dist is a NxM matrix where dist[i,j] is the norm
           between x[i,:] and y[j,:]
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||
    '''
    differences = x.unsqueeze(1) - y.unsqueeze(0)
    distances = torch.sum(differences**2, -1).sqrt()
    return distances



class WeightedHausdorffDistanceNotWorking(nn.Module):
    def __init__(self,
                 dis_matrix):
        """
        :param resized_height: Number of rows in the image.
        :param resized_width: Number of columns in the image.
        :param return_2_terms: Whether to return the 2 terms
                               of the WHD instead of their sum.
                               Default: False.
        :param device: Device where all Tensors will reside.

        Adapted from https://github.com/HaipengXiong/weighted-hausdorff-loss/blob/master/object-locator/losses.py#L154
        @article{whd-loss,
          title={Weighted Hausdorff Distance: A Loss Function For Object Localization},
          author={J. Ribera and D. G{\"u}era and Y. Chen and E. Delp},
          journal={arXiv:1806.07564},
          month={June},
          year={2018}
        }
        """
        super().__init__()
        # Prepare all possible (row, col) locations in the image
        self.dis_matrix = dis_matrix
        self.max_dist = torch.max(dis_matrix)

    def forward(self, prob_map, gt):
        """
        Compute the Weighted Hausdorff Distance function
         between the estimated probability map and ground truth points.
        The output is the WHD averaged through all the batch.

        :param prob_map: (B x NVoxels x C ) Tensor of the probability map of the estimation.
                         B is batch size, NVoxel is the number of voxels and C the channel
                         Values must be between 0 and 1. ?
        :param gt: List of Tensors of the Ground Truth points.
                   Must be of size B as in prob_map.
                   Each element in the list must be a 2D Tensor,
                   where each row is the (y, x), i.e, (row, col) of a GT point.
        :return: Single-scalar Tensor with the Weighted Hausdorff Distance.
                 If self.return_2_terms=True, then return a tuple containing
                 the two terms of the Weighted Hausdorff Distance.
        """

        if prob_map.device != self.dis_matrix.device:
            self.dis_matrix = self.dis_matrix.to(prob_map.device)
        assert prob_map.dim() == 3, 'The probability map must be( B x NVoxels x C)'

        batch_size = prob_map.shape[0]
        assert batch_size == len(gt)
        flatten_prob_map = torch.linalg.norm(prob_map, dim=2)
        flatten_prob_map = flatten_prob_map
        terms_1 = []
        terms_2 = []
        for b in range(batch_size):
            # One by one
            prob_map_b = flatten_prob_map[b] / torch.max(flatten_prob_map[b]).detach()
            gt_b = gt[b]

            # Pairwise distances between all possible locations and the GTed locations
            n_gt_pts = gt_b.size()[0]
            d_matrix = self.dis_matrix[:, gt_b]

            # Reshape probability map as a long column vector,
            # and prepare it for multiplication
            n_est_pts = prob_map_b.sum()
            eps = 1e-6
            alpha = -1

            # Weighted Hausdorff Distance
            term_1 = (1. / (n_est_pts + eps)) * \
                     torch.sum(prob_map_b * torch.min(d_matrix, 1)[0])

            inner_term2 = prob_map_b[:, None] * d_matrix + ((1 - prob_map_b) * self.max_dist)[:,None]
            term_2 = torch.mean(torch.mean((inner_term2+eps)**alpha, dim=0) ** (1 / alpha))

            if False:
                #I Do not understand this code!!
                # Reshape probability map as a long column vector,
                # and prepare it for multiplication
                p = prob_map_b.view(prob_map_b.nelement())
                p_replicated = p.view(-1, 1).repeat(1, n_gt_pts)

                d_div_p = torch.min((d_matrix + eps) /
                                    (p_replicated**alpha + eps / self.max_dist), 0)[0]
                d_div_p = torch.clamp(d_div_p, 0, self.max_dist)
                original_term_2 = torch.mean(d_div_p, 0)
                # terms_1[b] = term_1
                # terms_2[b] = term_2
            terms_1.append(term_1)
            terms_2.append(term_2)

        terms_1 = torch.stack(terms_1)
        terms_2 = torch.stack(terms_2)

        res = terms_1.mean() + terms_2.mean()

        return res

class WeightedHausdorffDistance(nn.Module):
    """
    Copyright &copyright © (c) 2019 The Board of Trustees of Purdue University and the Purdue Research Foundation.
    All rights reserved.

    This software is covered by US patents and copyright.
    This source code is to be used for academic research purposes only, and no commercial use is allowed.

    For any questions, please contact Edward J. Delp (ace@ecn.purdue.edu) at Purdue University.

    Last Modified: 10/02/2019

    __license__ = "CC BY-NC-SA 4.0"
    __authors__ = "Javier Ribera, David Guera, Yuhao Chen, Edward J. Delp"
    __version__ = "1.6.0"
    """

    def __init__(self, dis_matrix, p=-1):
        """
        :param dis_matrix: Distance matrix between all possible set of points
        :param p: Scaling factor for generalized mean, default = -1

        @article{whd-loss,
          title={Weighted Hausdorff Distance: A Loss Function For Object Localization},
          author={J. Ribera and D. G{\"u}era and Y. Chen and E. Delp},
          journal={arXiv:1806.07564},
          month={June},
          year={2018}
        }
        https://github.com/javiribera/locating-objects-without-bboxes/blob/master/object-locator/losses.py
        """
        super().__init__()

        self.dis_matrix = dis_matrix
        self.max_dist = torch.max(dis_matrix)
        self.p = p

    def forward(self, prob_map, gt):
        """
        Compute the Weighted Hausdorff Distance function
        between the estimated probability map and ground truth points.
        The output is the WHD averaged through all the batch.

        :param prob_map: (B x H x W) Tensor of the probability map of the estimation.
                         B is batch size, H is height and W is width.
                         Values must be between 0 and 1.
        :param gt: List of Tensors of the Ground Truth points.
                   Must be of size B as in prob_map.
                   Each element in the list must be a 2D Tensor,
                   where each row is the (y, x), i.e, (row, col) of a GT point.
        :return: Single-scalar Tensor with the Weighted Hausdorff Distance.
        """
        if prob_map.dim() == 3:
            prob_map = torch.linalg.norm(prob_map, dim=2)

        if prob_map.device != self.dis_matrix.device:
            self.dis_matrix = self.dis_matrix.to(prob_map.device)
        assert prob_map.dim() == 2, 'The probability map must be (B x Positions)'
        prob_map = (prob_map-prob_map.min().detach()) / (prob_map.max()-prob_map.min()).detach()
        prob_map = torch.clamp(prob_map, 0, 1) #Probability only between 0 and 1 allowed!
        batch_size = prob_map.shape[0]
        assert batch_size == len(gt)

        terms_1 = []
        terms_2 = []
        for b in range(batch_size):

            # One by one
            prob_map_b = prob_map[b, :]
            gt_b = gt[b]
            # Pairwise distances between all possible locations and the GTed locations
            n_gt_pts = gt_b.size()[0]


            p = prob_map_b.view(prob_map_b.nelement())
            n_est_pts = p.sum()
            p_replicated = p.view(-1, 1).repeat(1, n_gt_pts)

            # Weighted Hausdorff Distance
            term_1 = (1 / (n_est_pts + 1e-6)) * \
                     torch.sum(p * torch.min(self.dis_matrix[:,gt_b], 1)[0])
            weighted_d_matrix = (1 - p_replicated) * self.max_dist + p_replicated * self.dis_matrix[:, gt_b]
            minn = generaliz_mean(weighted_d_matrix,
                                  p=self.p,
                                  dim=0, keepdim=False)
            term_2 = torch.mean(minn)

            # terms_1[b] = term_1
            # terms_2[b] = term_2
            terms_1.append(term_1)
            terms_2.append(term_2)

        terms_1 = torch.stack(terms_1)
        terms_2 = torch.stack(terms_2)


        res = terms_1.mean() + terms_2.mean()

        return res

def generaliz_mean(tensor, dim, p=-9, keepdim=False):
    # """
    # Computes the softmin along some axes.
    # Softmin is the same as -softmax(-x), i.e,
    # softmin(x) = -log(sum_i(exp(-x_i)))

    # The smoothness of the operator is controlled with k:
    # softmin(x) = -log(sum_i(exp(-k*x_i)))/k

    # :param input: Tensor of any dimension.
    # :param dim: (int or tuple of ints) The dimension or dimensions to reduce.
    # :param keepdim: (bool) Whether the output tensor has dim retained or not.
    # :param k: (float>0) How similar softmin is to min (the lower the more smooth).
    # """
    # return -torch.log(torch.sum(torch.exp(-k*input), dim, keepdim))/k
    """
    The generalized mean. It corresponds to the minimum when p = -inf.
    https://en.wikipedia.org/wiki/Generalized_mean
    :param tensor: Tensor of any dimension.
    :param dim: (int or tuple of ints) The dimension or dimensions to reduce.
    :param keepdim: (bool) Whether the output tensor has dim retained or not.
    :param p: (float<0).
    """
    assert p < 0
    res = torch.mean((tensor + 1e-6) ** p, dim, keepdim=keepdim) ** (1. / p)
    return res

def activation_focal_loss(pred, gt):
    """
    Compute the focal loss between the predicted and ground truth values
    Everything above a threshold is considered a positive class
    """
    eps = 1e-6
    nonzero_activ = (gt.abs() > eps).float()
    pred_prob = pred.abs() / (pred.abs().max().detach() + eps)
    focal_loss = torchvision.ops.sigmoid_focal_loss(pred_prob, nonzero_activ, alpha=0.25, gamma=2.0, reduction='sum')

    return focal_loss



def activation_weighted_mse(pred, gt, zero_scale=0.01):
    """
    Compute the MSE between the predicted and ground truth values,
    However, we use two weights for balancing zero and non-zero voxels
    """
    eps = 1e-6
    nonzero_activ = (gt.abs() > eps).float()
    zero_activ = (gt.abs() <= eps).float()
    if zero_scale is None:

        # We weight dynamically by the percentage of voxels in each class
        bs, c, *spatial = gt.shape
        nonzero_activ = nonzero_activ.view(bs,-1)
        zero_activ = zero_activ.view(bs, -1)
        n_voxels = zero_activ.shape[1]
        weight = (nonzero_activ * ((nonzero_activ.sum(1, keepdims=True) + eps)/n_voxels) +
                  zero_activ * ((zero_activ.sum(1, keepdims=True) + eps)/n_voxels))
        weight = weight.view(bs, c, *spatial)
    else:
        weight = nonzero_activ + zero_scale * zero_activ
    mse = torch.sum(weight * (pred - gt) ** 2)
    return mse


def activation_weighted_mae(pred, gt, zero_scale=0.01):
    """
    Compute the MSE between the predicted and ground truth values,
    However, we use two weights for balancing zero and non-zero voxels
    """
    eps = 1e-6
    nonzero_activ = (gt.abs() > eps).float()
    zero_activ = (gt.abs() <= eps).float()
    if zero_scale is None:

        # We weight dynamically by the percentage of voxels in each class
        bs, c, *spatial = gt.shape
        nonzero_activ = nonzero_activ.view(bs,-1)
        zero_activ = zero_activ.view(bs, -1)
        n_voxels = zero_activ.shape[1]
        weight = (nonzero_activ * ((nonzero_activ.sum(1, keepdims=True) + eps)/n_voxels) +
                  zero_activ * ((zero_activ.sum(1, keepdims=True) + eps)/n_voxels))
        weight = weight.view(bs, c, *spatial)
    else:
        weight = nonzero_activ + zero_scale * zero_activ
    mse = torch.sum(weight * torch.abs(pred - gt))
    return mse



class PseudoHuberLoss(nn.Module):
    def __init__(self, reduction="mean", c=1.0):
        super(PseudoHuberLoss, self).__init__()
        self.c = c
        assert reduction in ["mean", "sum", "none"], f"Invalid reduction {reduction} for PseudoHuberLoss"
        self.reduction = reduction

    def forward(self, y_pred, y_true):
        loss = torch.sqrt(torch.square(y_true - y_pred) + self.c**2) - self.c
        if self.reduction == "mean":
            return torch.mean(loss)
        elif self.reduction == "sum":
            return torch.sum(loss)
        else:
            return loss

class MemoryRegularization:
    def __init__(self):
        self.memory_1 = None
        self.memory_2 = None
        self.current_memory = 0

    def memory_hook(self, module, input, output):
        if self.current_memory == 0:
            self.memory_1 = output
            self.current_memory = 1
        else:
            self.memory_2 = output
            self.current_memory = 0

    def clear_memory(self):
        self.memory_1 = None
        self.memory_2 = None
        self.current_memory = 0

