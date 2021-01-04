import torch
from torch.nn import Linear
from graphae.fully_connected import FNN


class Permuter(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.scoring_fc = Linear(input_dim, 1)
        self.soft_sort = SoftSort()

    def forward(self, x, mask, hard=False, tau=1.0):
        # x [batch_size, element_dim]
        scores = self.scoring_fc(x)
        fill_value = scores.min().item() - 1
        scores = scores.masked_fill(mask.unsqueeze(-1) == 0, fill_value)
        perm = self.soft_sort(scores.squeeze(), hard=hard, tau=tau)  # [batch_size, max_len, max_len]
        perm = perm.transpose(2, 1)
        return perm


class SoftSort(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, scores, hard=False, tau=1.0):
        scores = scores.unsqueeze(-1)
        sorted = scores.sort(descending=True, dim=1)[0]
        pairwise_diff = (scores.transpose(1, 2) - sorted).abs().neg() / tau
        perm = pairwise_diff.softmax(-1)

        if hard:
            perm_ = torch.zeros_like(perm, device=perm.device)
            perm_.scatter_(-1, perm.topk(1, -1)[1], value=1)
            perm = (perm_ - perm).detach() + perm
        return perm
