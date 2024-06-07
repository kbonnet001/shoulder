from neural_networks.data_preparation import data_preparation_create_tensor, compute_samples, plot_datas_distribution
from neural_networks.data_tranning import plot_loss_and_accuracy, plot_predictions_and_targets, train, evaluate
from neural_networks.Model import Model
from neural_networks.activation_functions import *
from neural_networks.MuscleDataset import MuscleDataset
import torch
from torch.utils.data import DataLoader, random_split
from neural_networks.EarlyStopping import EarlyStopping
import torch.optim as optim

def test_model_supervised_learning(filename) : 

    X_tensor, y_tensor = data_preparation_create_tensor(filename, 0 )
    dataset_muscle_PECM2 = MuscleDataset(X_tensor, y_tensor)

    train_val_size, test_size = compute_samples(dataset_muscle_PECM2, 0.80)
    train_val_dataset, test_dataset = random_split(dataset_muscle_PECM2, [train_val_size, test_size]) #450 + 50

    train_size, val_size = compute_samples(train_val_dataset, 0.80)
    train_dataset, val_dataset = random_split(train_val_dataset, [train_size, val_size]) #405 + 45
    
    plot_datas_distribution(X_tensor, y_tensor)

    # Create DataLoaders
    batch_size = 64

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle = True) #13
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle = False) #2
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle = False) #2

    # Définir le modèle
    input_size = 4  # Nombre de features en entrée
    output_size = 1  # Nombre de sorties (dans ce cas, une seule valeur de prédiction)
    n_layers = 1
    n_nodes = 4
    # activation = Swish()
    activation = nn.ReLU()
    L1_penalty = 0.001
    L2_penalty = 0.001
    use_batch_norm = True
    learning_rate = 1e-4

    n_epochs = 1000

    model = Model(input_size, output_size, activation, L1_penalty, L2_penalty, use_batch_norm, 0.5)

    optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    criterion = nn.MSELoss()
    # criterion = CustomLoss()

    # Entraînement du modèle
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    # Initialiser le planificateur ReduceLROnPlateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=20, min_lr=1e-8, verbose=True)

    # Initialiser l'arrêt précoce
    early_stopping = EarlyStopping(monitor='val_mae', patience=50, min_delta=0.00001, verbose=True)

    for epoch in range(n_epochs):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion)
        val_loss, val_acc = evaluate(model, val_loader, criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f'Epoch [{epoch+1}/{n_epochs}], Train Loss: {train_losses[-1]:.6f}, Val Loss: {val_losses[-1]:.6f}, Train Acc: {train_acc:.6f}, Val Acc: {val_acc:.6f}')

        # Réduire le taux d'apprentissage si nécessaire
        scheduler.step(val_losses[-1])

        # Vérifier l'arrêt précoce
        early_stopping(val_losses[-1])
        if early_stopping.early_stop:
            print("Early stopping at epoch:", epoch+1)
            break

    # Évaluation du modèle sur l'ensemble de test
    test_loss, test_acc = evaluate(model, test_loader, criterion)
    print(f'Test Loss: {test_loss:.6f}, Test Acc: {test_acc:.6f}')
    
    plot_loss_and_accuracy(train_losses, val_losses, train_accs, val_accs)