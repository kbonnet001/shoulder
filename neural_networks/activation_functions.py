import torch
import torch.nn as nn
import torch.nn.functional as F

class Swish(nn.Module):
    """
    Custom implementation of the Swish activation function.

    The Swish function is defined as f(x) = x * sigmoid(x), where sigmoid(x) = 1 / (1 + exp(-x)).
    It is a smooth, non-monotonic function that can improve the training of deep neural networks.
    """

    def forward(self, x):
        """
        Forward pass of the Swish activation function.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the Swish function.
        """
        # Apply the Swish activation function
        return x * torch.sigmoid(x)
