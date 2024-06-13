from neural_networks.data_preparation import data_preparation_create_tensor, compute_samples, plot_datas_distribution
from neural_networks.data_tranning import plot_loss_and_accuracy, plot_predictions_and_targets, train, evaluate
from neural_networks.Model import Model
from neural_networks.activation_functions import *
from neural_networks.MuscleDataset import MuscleDataset
import torch
from torch.utils.data import DataLoader, random_split
from neural_networks.EarlyStopping import EarlyStopping
import torch.optim as optim
import os
import json

def prepare_data(batch_size, filename, plot=False) : 
    
    X_tensor, y_tensor = data_preparation_create_tensor(filename, 0)
    dataset_muscle_PECM2 = MuscleDataset(X_tensor, y_tensor)

    train_val_size, test_size = compute_samples(dataset_muscle_PECM2, 0.80)
    train_val_dataset, test_dataset = random_split(dataset_muscle_PECM2, [train_val_size, test_size]) #450 + 50

    train_size, val_size = compute_samples(train_val_dataset, 0.80)
    train_dataset, val_dataset = random_split(train_val_dataset, [train_size, val_size]) #405 + 45
    
    if plot : 
        plot_datas_distribution(X_tensor, y_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle = True) #13
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle = False) #2
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle = False) #2
    
    input_size = len(train_dataset[0][0])
    output_size = 1 
    
    return train_loader, val_loader, test_loader, input_size, output_size

def train_model_supervised_learning(train_loader, val_loader, test_loader, input_size, output_size, activation, n_layers, n_nodes, L1_penalty, L2_penalty, use_batch_norm, dropout_prob, learning_rate, n_epochs, file_path) : 

    model = Model(input_size, output_size, activation, n_layers, n_nodes, L1_penalty, L2_penalty, use_batch_norm, dropout_prob)

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
    
    plot_predictions_and_targets(model, train_loader, "Train loader", 100)
    plot_predictions_and_targets(model, val_loader, "Validation loader", 100)
    plot_predictions_and_targets(model, test_loader, "Test loader", 100)
    
    
    # Sauvegarder la configuration du modèle
    config = {
        'input_shape': input_size,
        'output_shape': output_size,
        'activation': 'ReLU',  # Sauvegarder le nom de l'activation
        'n_layers' : n_layers,
        'n_nodes' : n_nodes,
        'L1_penalty': L1_penalty,
        'L2_penalty': L2_penalty,
        'use_batch_norm': use_batch_norm,
        'dropout_prob': output_size
    }

    with open('model_config.json', 'w') as f:
        json.dump(config, f)
    torch.save(model.state_dict(), file_path)
    
def visualize_prediction(train_loader, val_loader, test_loader, file_path) : 
    
    # Charger la configuration du modèle
    with open('model_config.json', 'r') as f:
        config = json.load(f)

    # Recréer l'activation
    activation = getattr(nn, config['activation'])()

    # Recréer le modèle avec la configuration chargée
    model = Model(
        input_shape=config['input_shape'],
        output_shape=config['output_shape'],
        activation=activation,
        n_layers=config['n_layers'], 
        n_nodes=config['n_nodes'],
        L1_penalty=config['L1_penalty'],
        L2_penalty=config['L2_penalty'],
        use_batch_norm=config['use_batch_norm'],
        dropout_prob=config['dropout_prob']
    )

    model.load_state_dict(torch.load(file_path))
    model.eval()
    
    plot_predictions_and_targets(model, train_loader, "Train loader", 100)
    plot_predictions_and_targets(model, val_loader, "Validation loader", 100)
    plot_predictions_and_targets(model, test_loader, "Test loader", 100)
    
def main_superised_learning(filename, retrain, file_path) : 

    batch_size = 128
    train_loader, val_loader, test_loader, input_size, output_size = prepare_data(batch_size, filename)
    
    if retrain or os.path.exists(file_path) == False: 
        
        n_layers = 1
        n_nodes = [10]
        
        activation = nn.ReLU()
        L1_penalty = 0.001
        L2_penalty = 0.001
        use_batch_norm = True
        dropout_prob = 0.5
        learning_rate = 1e-4

        n_epochs = 1000
        
        train_model_supervised_learning(train_loader, val_loader, test_loader, input_size, output_size, activation, n_layers, n_nodes, L1_penalty, L2_penalty, use_batch_norm, dropout_prob, learning_rate, n_epochs, file_path)
        
    visualize_prediction(train_loader, val_loader, test_loader, file_path)
        