import torch
from torch import nn

class SSBayesRiskLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        """误差+方差"""
        super().__init__(*args, **kwargs)

    def forward(self, evidences, labels):
        alphas = evidences + 1.0
        strength = torch.sum(alphas, dim=-1, keepdim=True)
        probabilities = alphas / strength

        error = (labels - probabilities) ** 2
        variance = probabilities * (1.0 - probabilities) / (strength + 1.0)

        loss = torch.sum(error + variance, dim=-1, keepdim=True)

        return torch.mean(loss)


class KLDivergenceLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        """将无法正确分类的样本证据缩小到零"""
        super().__init__(*args, **kwargs)

    def forward(self, evidences, labels):
        num_classes = evidences.size(-1)
        alphas = evidences + 1.0
        alphas_tilde = labels + (1.0 - labels) * alphas
        strength_tilde = torch.sum(alphas_tilde, dim=-1, keepdim=True)

        # lgamma是gamma函数的对数
        first_term = (
            torch.lgamma(strength_tilde)
            - torch.lgamma(evidences.new_tensor(num_classes, dtype=torch.float32))
            - torch.sum(torch.lgamma(alphas_tilde), dim=-1, keepdim=True)
        )
        second_term = torch.sum(
            (alphas_tilde - 1.0) * (torch.digamma(alphas_tilde) - torch.digamma(strength_tilde)), dim=-1, keepdim=True
        )
        loss = torch.mean(first_term + second_term)

        return loss
