import torch.nn as nn

class Model(nn.Module):
    def __init__(self, input_size, output_size, n_nodes, activations, L1_penalty, L2_penalty, use_batch_norm, dropout_prob):
        super(Model, self).__init__()
        layers = []
        in_features = input_size
        for i in range(len(n_nodes)):
            layers.append(nn.Linear(in_features, n_nodes[i]))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(n_nodes[i]))
            layers.append(activations[i])
            layers.append(nn.Dropout(dropout_prob))
            in_features = n_nodes[i]
        layers.append(nn.Linear(in_features, output_size))
        self.model = nn.Sequential(*layers)
        self.L1_penalty = L1_penalty
        self.L2_penalty = L2_penalty

    def forward(self, x):
        return self.model(x)