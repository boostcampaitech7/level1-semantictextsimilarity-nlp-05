import torch
import torch.nn as nn


class PearsonCorrelationLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predictions, targets):
        x = predictions.view(-1)
        y = targets.view(-1)

        vx = x - torch.mean(x)
        vy = y - torch.mean(y)

        cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
        return 1 - cost

class CombinedMSEPearsonLoss(nn.Module):
    def __init__(self, alpha=0.7):
        super().__init__()
        self.alpha = alpha
        self.MSE_loss = nn.MSELoss()
        self.pearson_loss = PearsonCorrelationLoss()

    def forward(self, predictions, targets):
        mse = self.MSE_loss(predictions, targets)
        pearson = self.pearson_loss(predictions, targets)
        return self.alpha * mse + (1 - self.alpha) * pearson
    
class CombinedL1PearsonLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.L1_loss = nn.L1Loss()
        self.pearson_loss = PearsonCorrelationLoss()

    def forward(self, predictions, targets):
        l1 = self.L1_loss(predictions, targets)
        pearson = self.pearson_loss(predictions, targets)
        return self.alpha * l1 + (1 - self.alpha) * pearson
######################