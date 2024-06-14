import torch
import torch.nn as nn

class ModifiedHuberLoss(nn.Module):
    # Example usage
    # criterion = ModifiedHuberLoss(delta=1.0, factor=1.5)
    def __init__(self, delta=1.0, factor=1.5):
        super(ModifiedHuberLoss, self).__init__()
        self.delta = delta
        self.factor = factor

    def forward(self, y_pred, y_true):
        error = y_true - y_pred
        abs_error = torch.abs(error)
        delta_tensor = torch.tensor(self.delta, dtype=abs_error.dtype, device=abs_error.device)
        quadratic = torch.min(abs_error, delta_tensor)
        linear = abs_error - quadratic
        loss = 0.5 * quadratic**2 + delta_tensor * linear
        return torch.mean(loss * (1 + self.factor * abs_error))

class LogCoshLoss(nn.Module):
    # Example usage
    # criterion = LogCoshLoss(factor=1.5)
    def __init__(self, factor=1.5):
        super(LogCoshLoss, self).__init__()
        self.factor = factor

    def forward(self, y_pred, y_true):
        error = y_true - y_pred
        logcosh = torch.log(torch.cosh(error))
        return torch.mean(logcosh * (1 + self.factor * torch.abs(error)))

class ExponentialLoss(nn.Module):
    # Example usage
    # criterion = ExponentialLoss(alpha=0.5)
    def __init__(self, alpha=0.5):
        super(ExponentialLoss, self).__init__()
        self.alpha = alpha

    def forward(self, y_pred, y_true):
        error = torch.abs(y_true - y_pred)
        loss = torch.exp(self.alpha * error) - 1
        return torch.mean(loss)




