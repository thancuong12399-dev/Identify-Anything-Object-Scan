import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ArcMarginProduct(nn.Module):
    """
    Additive Angular Margin Loss (ArcFace) Layer.
    Reference: Deng et al. 'ArcFace: Additive Angular Margin Loss for Deep Face Recognition' (CVPR 2019)
    """
    def __init__(self, in_features: int, out_features: int, s: float = 64.0, m: float = 0.50):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m

        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.threshold = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        # L2-normalize input features and class weights to unit hypersphere
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.clamp(cosine ** 2, 0.0, 1.0))

        # cos(theta + m) = cos(theta)*cos(m) - sin(theta)*sin(m)
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.threshold, phi, cosine - self.mm)

        # One-hot encoding of target classes
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1.0)

        # Scale logits by hypersphere radius s
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output
