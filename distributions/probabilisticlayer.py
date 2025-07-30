import torch
import torch.nn as nn
import torch.distributions as dist

from distributions import pointdistribution


class Probabilistic_Layer(nn.Module):
    def __init__(self, distribution: str = "none", in_features: int = 1000, use_bias: bool = True, eps: float = 1e-4):
        super().__init__()

        self.dim = in_features
        self.distribution = distribution

        if distribution in [
            "none",
            "point",
            "sphere",
            "vonMisesFisherNorm",
            "sphereNoFC",
            "powerspherical_wo_fc_reuse",
        ]:
            out_features = in_features

        elif distribution == "powerspherical":
            out_features = in_features + 1
        elif distribution in ["powerspherical_wo_fc_lin", "powerspherical_wo_fc_nonlin"]:
            out_features = 1
        elif distribution == "normal" or "vade" in distribution or "entropyregnormal" == distribution:
            out_features = in_features * 2
        elif distribution == "vonMisesFisherNode" or distribution == "normalSingleScale":
            out_features = in_features + 1
        elif "MixtureOfGaussians" in distribution:
            self.n_experts = int("".join(filter(str.isdigit, distribution)))
            out_features = in_features * 3 * self.n_experts
        else:
            raise NotImplementedError(
                f"Distribution {distribution} not implemented yet. Choose from "
                '["none", "point", "normal","powerspherical","sphere","vonMisesFisherNorm", '
                '"MixtureOfGaussians", "vonMisesFisherNode", "MixtureOfGaussians","entropyregnormal"]'
            )

        if distribution not in ["none", "sphereNoFC", "powerspherical_wo_fc_reuse", "powerspherical_wo_fc_nonlin"]:
            self.layer = nn.Linear(in_features, out_features, bias=use_bias)
        elif distribution == "powerspherical_wo_fc_nonlin":
            hidden_dim = in_features // 2
            self.layer = nn.Sequential(
                nn.Linear(in_features, hidden_dim, bias=use_bias),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, out_features, bias=use_bias),
            )
        else:  # For none use nn.Identity
            self.layer = nn.Identity()
        self.eps = eps

    def forward(self, x):

        if self.distribution == "normal" or "vade" in self.distribution or "entropyregnormal" == self.distribution:
            mean, logvar = torch.chunk(self.layer(x), 2, dim=1)
            std = nn.functional.softplus(logvar) + self.eps

            out_dist = dist.Normal(mean, std)

            return out_dist
        if self.distribution == "normalSingleScale":
            y = self.layer(x)
            mean = y[:, :-1]
            std = nn.functional.softplus(y[:, -1:]) + self.eps
            return dist.Normal(mean, std)

        if self.distribution == "none" or self.distribution == "point":
            return pointdistribution.PointDistribution(self.layer(x))

        if self.distribution == "sphere" or self.distribution == "sphereNoFC":
            mu = self.layer(x)
            norm_mu = torch.linalg.norm(mu, dim=1, keepdim=True)
            return pointdistribution.PointDistribution(mu / norm_mu)

        if self.distribution == "powerspherical":
            feats = self.layer(x)
            mu = nn.functional.normalize(feats[:, : self.dim], dim=1)
            # norm_mu = torch.linalg.norm(mu, dim=1, keepdim=True)
            const = torch.pow(torch.tensor(self.dim).to(mu.device), 1 / 2.0)
            kappa = const * nn.functional.softplus(feats[:, -1]) + self.eps
            return powerspherical.PowerSpherical(mu, kappa)

        if self.distribution in ["powerspherical_wo_fc_lin", "powerspherical_wo_fc_nonlin"]:
            mu = nn.functional.normalize(x, dim=1)
            # norm_mu = torch.linalg.norm(mu, dim=1, keepdim=True).detach()
            const = torch.pow(torch.tensor(self.dim).to(mu.device), 1 / 2.0)
            kappa = self.layer(x)
            kappa = const * nn.functional.softplus(kappa[:, 0]) + self.eps

            if torch.sum(torch.isnan(kappa)):
                print("There is somewhere a nan in Kappa!", torch.sum(torch.isnan(kappa)))
                print(kappa)
                print(self.layer(x))
                print(mu)
            # assert( (mu / norm_mu == nn.functional.normalize(mu, dim=1)).all())
            return powerspherical.PowerSpherical(mu, kappa)

        if self.distribution == "powerspherical_wo_fc_reuse":
            feats = x
            padding = torch.zeros((feats.shape[0], 1), device=feats.device)
            mu = torch.cat([feats[:, : self.dim - 1], padding], dim=-1)
            norm_mu = torch.linalg.norm(mu, dim=1, keepdim=True)
            const = torch.pow(torch.tensor(self.dim).to(mu.device), 1 / 2.0)
            kappa = const * nn.functional.softplus(feats[:, -1]) + self.eps
            return powerspherical.PowerSpherical(mu / norm_mu, kappa)

        if self.distribution == "vonMisesFisherNorm":
            mu = self.layer(x)
            norm_mu = torch.linalg.norm(mu, dim=1, keepdim=True)

            return vonmisesfisher.VonMisesFisher(mu / norm_mu, torch.pow(norm_mu, 2))

        if self.distribution == "vonMisesFisherNode":
            mu = self.layer(x)
            norm_mu = torch.linalg.norm(mu[:, :-1], dim=1, keepdim=True)
            kappa = nn.functional.softplus(mu[:, -1]) + self.eps

            return vonmisesfisher.VonMisesFisher(mu[:, :-1] / norm_mu, torch.pow(kappa, 2))

        if "MixtureOfGaussians" in self.distribution:
            mu, log_var, pi = torch.chunk(self.layer(x), 3, dim=1)
            std = nn.functional.softplus(log_var)
            Mixture = dist.categorical.Categorical(logits=pi.view(x.shape[0], -1, self.n_experts))
            Component = dist.Normal(
                mu.view(x.shape[0], -1, self.n_experts),
                std.view(x.shape[0], -1, self.n_experts),
            )
            return MixtureOfExperts.MixtureOfExperts(Mixture, Component)

        raise ValueError("Forward pass with unknown distribution.")
