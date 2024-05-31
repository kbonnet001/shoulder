from data_preparation import data_preparation_create_tensor, compute_samples, plot_datas_distribution
from data_tranning import plot_loss_and_accuracy, plot_predictions_and_targets, train, evaluate
from CustomModel import CustomModel
from activation_functions import *
from MuscleDataset import MuscleDataset
import torch
from torch.utils.data import DataLoader, random_split


X_tensor, y_tensor = data_preparation_create_tensor("df_PECM2_datas.xlsx")
plot_datas_distribution(X_tensor, y_tensor) 
dataset_muscle_PECM2 = MuscleDataset(X_tensor, y_tensor)

train_val_size, test_size = compute_samples(dataset_muscle_PECM2, 0.90)
train_val_dataset, test_dataset = random_split(dataset_muscle_PECM2, [train_val_size, test_size]) #450 + 50

train_size, val_size = compute_samples(train_val_dataset, 0.90)
train_dataset, val_dataset = random_split(train_val_dataset, [train_size, val_size]) #405 + 45

# Create DataLoaders
batch_size = 64

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle = True) #13
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle = False) #2
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle = False) #2

# Définir le modèle
input_size = len(X_tensor[0])  
output_size = 1  
n_layers = 3
n_nodes = 256
activation = Swish()
L1_penalty = 0.00
L2_penalty = 0.001
use_batch_norm = True
learning_rate = 1e-6

n_epochs = 5000 #1000 dans les papiers mais n'évolue plus au delà de 500

model = CustomModel(input_size, output_size, n_layers, n_nodes, activation, L1_penalty, L2_penalty, use_batch_norm)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

# Entraînement du modèle
train_losses = []
val_losses = []
train_accs = []
val_accs = []

for epoch in range(n_epochs):
    train_loss, train_acc = train(model, train_loader, optimizer, criterion)
    val_loss, val_acc = evaluate(model, val_loader, criterion)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    print(f'Epoch [{epoch+1}/{n_epochs}], Train Loss: {train_losses[-1]:.6f}, Val Loss: {val_losses[-1]:.6f}, Train Acc: {train_acc:.6f}, Val Acc: {val_acc:.6f}')

# Évaluation du modèle sur l'ensemble de test
test_loss, test_acc = evaluate(model, test_loader, criterion)
print(f'Test Loss: {test_loss:.6f}, Test Acc: {test_acc:.6f}')

plot_loss_and_accuracy(train_losses, val_losses, train_accs, val_accs)

plot_predictions_and_targets(model, test_loader, 100)
