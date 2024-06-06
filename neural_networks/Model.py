import torch.nn as nn

class Model(nn.Module):
    def __init__(self, input_shape, output_shape, n_layers, n_nodes, activation, L1_penalty, L2_penalty, use_batch_norm):
        super(Model, self).__init__()
        layers = []
        for i in range(n_layers):
            layers.append(nn.Linear(input_shape if i == 0 else n_nodes, n_nodes))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(n_nodes))
            layers.append(activation)
        layers.append(nn.Linear(n_nodes, output_shape))
        self.model = nn.Sequential(*layers)

        self.L1_penalty = L1_penalty
        self.L2_penalty = L2_penalty

    def forward(self, x):
        output = self.model(x)
        return output