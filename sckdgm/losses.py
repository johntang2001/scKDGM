import torch
import torch.nn as nn
import torch.nn.functional as F


def info_nce_loss(z1, z2, temperature=0.7):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    logits = z1 @ z2.T / temperature
    labels = torch.arange(z1.shape[0], device=z1.device)
    return F.cross_entropy(logits, labels)


class NB(nn.Module):
    """Negative binomial negative log-likelihood."""

    def __init__(self, scale_factor=1.0, eps=1e-10):
        super().__init__()
        self.scale_factor = scale_factor
        self.eps = eps

    def forward(self, theta, y_true, y_pred, mean=True):
        eps = self.eps
        y_true = y_true.float()
        y_pred = torch.clamp(y_pred.float() * self.scale_factor, min=eps, max=1e6)
        theta = torch.clamp(theta.float(), min=eps, max=1e6)

        loss = (
            torch.lgamma(theta + eps)
            + torch.lgamma(y_true + 1.0)
            - torch.lgamma(y_true + theta + eps)
            + (theta + y_true) * torch.log1p(y_pred / (theta + eps))
            + y_true * (torch.log(theta + eps) - torch.log(y_pred + eps))
        )
        loss = torch.nan_to_num(loss, nan=float("inf"), posinf=float("inf"))
        return loss.mean() if mean else loss


class ZINB(nn.Module):
    """Zero-inflated negative binomial negative log-likelihood."""

    def __init__(self, ridge_lambda=0.0, scale_factor=1.0, eps=1e-10):
        super().__init__()
        self.ridge_lambda = ridge_lambda
        self.scale_factor = scale_factor
        self.eps = eps
        self.nb = NB(scale_factor=scale_factor, eps=eps)

    def forward(self, pi, theta, y_true, y_pred, mean=True):
        eps = self.eps
        y_true = y_true.float()
        y_pred = torch.clamp(y_pred.float() * self.scale_factor, min=eps, max=1e6)
        theta = torch.clamp(theta.float(), min=eps, max=1e6)
        pi = torch.clamp(pi.float(), min=eps, max=1.0 - eps)

        nb_nll = self.nb(theta, y_true, y_pred, mean=False)
        nb_case = nb_nll - torch.log(1.0 - pi + eps)
        zero_nb = torch.exp(theta * (torch.log(theta + eps) - torch.log(theta + y_pred + eps)))
        zero_case = -torch.log(pi + (1.0 - pi) * zero_nb + eps)
        loss = torch.where(y_true < 1e-8, zero_case, nb_case)
        if self.ridge_lambda > 0:
            loss = loss + self.ridge_lambda * pi.square()
        loss = torch.nan_to_num(loss, nan=float("inf"), posinf=float("inf"))
        return loss.mean() if mean else loss
