import torch.nn as nn
import torch
from distributions.pointdistribution import PointDistribution
class ClassicalApproach(nn.Module):
    # A wrapper class for the classical approach
    def __init__(self, approach='lasso', forward_model=None, hyperparam=None):
        super().__init__()
        self.approach = approach
        self.forward_model = forward_model
        self.hyperparam= hyperparam
        self.dummy_param = nn.Parameter(nn.Parameter(torch.zeros(1)))

    def set_hyperparam(self, hyperparam):
        self.hyperparam = hyperparam

    def forward(self, x, forward_index):
        if "LassoZero" in self.approach:
            out = self.forward_model.pytorch_lasso(x, forward_index=forward_index, max_iter=4000, united_pos=False,
                                                   lambda_lasso=150 if self.hyperparam is None else self.hyperparam, start_pseudo=False)
        elif "LassoPosZero" in self.approach:
            out = self.forward_model.pytorch_lasso(x, forward_index=forward_index, max_iter=4000, united_pos=True,
                                                   lambda_lasso=2682 if self.hyperparam is None else self.hyperparam,
                                                   start_pseudo=False)
                                                   #lambda_lasso=13894 if self.hyperparam is None else self.hyperparam, start_pseudo=False)
        elif "LassoPos" in self.approach:
            out = self.forward_model.pytorch_lasso(x, forward_index=forward_index, max_iter=8000, united_pos=True,
                                                   lambda_lasso=2000 if self.hyperparam is None else self.hyperparam)
        elif "Lasso_SciPy" in self.approach:
            out = self.forward_model.lasso_inverse(x, forward_index=forward_index, lambda_lasso=0.1 if self.hyperparam is None else self.hyperparam)
        elif "Lasso" in self.approach:
            out = self.forward_model.pytorch_lasso(x, forward_index=forward_index, max_iter=4000, lambda_lasso=150 if self.hyperparam is None else self.hyperparam)
        elif "Champagne" in self.approach:
            out = self.forward_model.champagne(x, forward_index=forward_index, alpha=0.05 if self.hyperparam is None else self.hyperparam)
        elif "GammaMap" in self.approach:
            out = self.forward_model.gamma_inverse(x, forward_index=forward_index, alpha=.2 if self.hyperparam is None else self.hyperparam)
        elif "eLORETA" in self.approach:
            out = self.forward_model.pseudo_inv_specific(x, forward_index=forward_index)
        elif self.approach in ["SupMNEeLORETA","SupMNE","SupdSPM","SupsLORETA", "SupMxNE", "Supbeamformer"]:

            out = self.forward_model.compute_mne_inverse(x, method=self.approach.replace("Sup",""))
        else:
            raise ValueError(f"Unknown approach: {self.approach}")
        return out
