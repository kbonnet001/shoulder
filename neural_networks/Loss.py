import torch
import torch.nn as nn

# In this file : 
# - ModifiedHuberLoss
# - LogCoshLoss
# - ExponentialLoss
#-------------------

class ModifiedHuberLoss(nn.Module):
    """ Modified Huber Loss function with an additional factor for scaling the loss based on absolute error.

    Args:
        delta (float): Threshold at which the loss function transitions from quadratic to linear. Default is 1.0.
        factor (float): Factor by which the loss is scaled based on the absolute error. Default is 1.5.
        
    Example: 
    criterion = ModifiedHuberLoss(delta=1.0, factor=1.5)
    """
    def __init__(self, delta=1.0, factor=1.5):
        super(ModifiedHuberLoss, self).__init__()
        self.delta = delta
        self.factor = factor

    def forward(self, y_pred, y_true):
        """ Compute the Modified Huber Loss between `y_pred` and `y_true`.

        Args:
            y_pred (Tensor): The predicted values.
            y_true (Tensor): The ground truth values.

        Returns:
            Tensor: The mean loss value.
        """
        error = y_true - y_pred
        abs_error = torch.abs(error)
        delta_tensor = torch.tensor(self.delta, dtype=abs_error.dtype, device=abs_error.device)
        quadratic = torch.min(abs_error, delta_tensor)
        linear = abs_error - quadratic
        loss = 0.5 * quadratic**2 + delta_tensor * linear
        return torch.mean(loss * (1 + self.factor * abs_error))

class LogCoshLoss(nn.Module):
    """ Log-Cosh Loss function with an optional scaling factor.

    Args:
        factor (float): Factor by which the loss is scaled based on the absolute error. Default is 1.5.
        
    Example:
    criterion = LogCoshLoss(factor=1.5)
    """
    
    def __init__(self, factor=1.5):
        super(LogCoshLoss, self).__init__()
        self.factor = factor

    def forward(self, y_pred, y_true):
        """ Compute the Log-Cosh Loss between `y_pred` and `y_true`.

        Args:
            y_pred (Tensor): The predicted values.
            y_true (Tensor): The ground truth values.

        Returns:
            Tensor: The mean loss value, scaled by the factor.
        """
        error = y_true - y_pred
        logcosh = torch.log(torch.cosh(error))
        return torch.mean(logcosh * (1 + self.factor * torch.abs(error)))

class ExponentialLoss(nn.Module):
    """ Exponential Loss function with a scaling parameter.

    Args:
        alpha (float): Scaling factor for the exponential term. Default is 0.5.
        
    Example:
    criterion = ExponentialLoss(alpha=0.5)
    """
    
    def __init__(self, alpha=0.5):
        super(ExponentialLoss, self).__init__()
        self.alpha = alpha

    def forward(self, y_pred, y_true):
        """ Compute the Exponential Loss between `y_pred` and `y_true`.

        Args:
            y_pred (Tensor): The predicted values.
            y_true (Tensor): The ground truth values.

        Returns:
            Tensor: The mean loss value.
        """
        error = torch.abs(y_true - y_pred)
        loss = torch.exp(self.alpha * error) - 1
        return torch.mean(loss)




