import torch
import torch.nn as nn
import torch.distributions as dist
from torch.distributions.utils import _standard_normal


class MixtureOfExperts(dist.Distribution):

    arg_constraints = {}
    def __init__(self, mixture_distribution, component_distribution):
        super().__init__(batch_shape=component_distribution.mean.size())
        # categorical Distribution for choosing
        # Gaussian Distribution for each domain
        self.mixture_distribution = mixture_distribution
        self.component_distribution = component_distribution
        self.n_experts = component_distribution.mean.shape[-1]

    def log_prob_loss(self, value):
        return -1 * torch.mean(self.log_prob(value))

    def log_prob(self, x):
        log_prob_x = self.component_distribution.log_prob(x.unsqueeze(-1))  # [S, B, k]
        log_mix_prob = torch.log_softmax(self.mixture_distribution.logits, dim=-1)  # [B, k]
        return torch.logsumexp(log_prob_x + log_mix_prob, dim=-1)  # [S, B]

    def rsample(self, sample_shape=torch.Size(), lambda_noise=1):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape[:-1], dtype=self.mean.dtype, device=self.mean.device)
        sampled_class = nn.functional.gumbel_softmax(self.mixture_distribution.logits, hard=True, dim=-1).expand(shape)
        return torch.sum(sampled_class * self.component_distribution.mean.expand(shape), dim=-1) + eps * torch.sum(
            (sampled_class * self.component_distribution.scale.expand(shape)), dim=-1
        )

    @property
    def variance(self):
        # Law of total variance: Var(Y) = E[Var(Y|X)] + Var(E[Y|X])
        probs = self.mixture_distribution.probs
        mean_cond_var = torch.sum(probs * self.component_distribution.variance, dim=-1)
        var_cond_mean = torch.sum(probs * (self.component_distribution.mean - self.mean.unsqueeze(-1)).pow(2.0), dim=-1)
        return mean_cond_var + var_cond_mean

    @property
    def scale(self):
        return self.variance**0.5

    @property
    def stddev(self):
        return self.variance ** 0.5

    @property
    def mean(self):
        probs = self.mixture_distribution.probs
        return torch.sum(probs * self.component_distribution.mean, dim=-1)

    @property
    def size(self):
        return self.mixture_distribution.mean.size[:-1]

    @property
    def shape(self):
        return self.mixture_distribution.mean.shape[:-1]


def move_expert_channel(x, n_experts):
    """Move the experts from the channel dimension to the last dim"""
    if len(x.shape) == 3:
        return x.view(x.shape[0], x.shape[1], -1, n_experts)

    return x.view(x.shape[0], n_experts, -1, *x.shape[2:]).movedim(1, -1)
