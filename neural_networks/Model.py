import torch.nn as nn

class Model(nn.Module):
    def __init__(self, input_shape, output_shape, activation, L1_penalty, L2_penalty, use_batch_norm, dropout_prob):
        super(Model, self).__init__()
        layers = []
        layers.append(nn.Linear(input_shape, 10))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(10))
        layers.append(activation)
        layers.append(nn.Dropout(dropout_prob))

        # layers.append(nn.Linear(6, 6))
        # if use_batch_norm:
        #     layers.append(nn.BatchNorm1d(6))
        # layers.append(activation)
        # # Ajout de la couche Dropout après chaque couche cachée
        # layers.append(nn.Dropout(dropout_prob))

        layers.append(nn.Linear(10, output_shape))
        self.model = nn.Sequential(*layers)

        self.L1_penalty = L1_penalty
        self.L2_penalty = L2_penalty

    def forward(self, x):
        output = self.model(x)
        return output