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
from neural_networks.save_model import *
from neural_networks.plot_visualisation import *

from neural_networks.Loss import *


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

# def train_model_supervised_learning(train_loader, val_loader, test_loader, input_size, output_size, activation, n_layers, n_nodes, L1_penalty, L2_penalty, use_batch_norm, dropout_prob, learning_rate, n_epochs, file_path) : 

#     model = Model(input_size, output_size, activation, n_layers, n_nodes, L1_penalty, L2_penalty, use_batch_norm, dropout_prob)

#     optimizer = torch.optim.Adam(model.parameters(), learning_rate)
#     criterion = nn.MSELoss()
#     # criterion = CustomLoss()

#     # Entraînement du modèle
#     train_losses = []
#     val_losses = []
#     train_accs = []
#     val_accs = []

#     # Initialiser le planificateur ReduceLROnPlateau
#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=20, min_lr=1e-8, verbose=True)

#     # Initialiser l'arrêt précoce
#     early_stopping = EarlyStopping(monitor='val_mae', patience=50, min_delta=0.00001, verbose=True)

#     for epoch in range(n_epochs):
#         train_loss, train_acc = train(model, train_loader, optimizer, criterion)
#         val_loss, val_acc = evaluate(model, val_loader, criterion)

#         train_losses.append(train_loss)
#         val_losses.append(val_loss)
#         train_accs.append(train_acc)
#         val_accs.append(val_acc)

#         print(f'Epoch [{epoch+1}/{n_epochs}], Train Loss: {train_losses[-1]:.6f}, Val Loss: {val_losses[-1]:.6f}, Train Acc: {train_acc:.6f}, Val Acc: {val_acc:.6f}')

#         # Réduire le taux d'apprentissage si nécessaire
#         scheduler.step(val_losses[-1])

#         # Vérifier l'arrêt précoce
#         early_stopping(val_losses[-1])
#         if early_stopping.early_stop:
#             print("Early stopping at epoch:", epoch+1)
#             break

#     # Évaluation du modèle sur l'ensemble de test
#     test_loss, test_acc = evaluate(model, test_loader, criterion)
#     print(f'Test Loss: {test_loss:.6f}, Test Acc: {test_acc:.6f}')
    
#     plot_loss_and_accuracy(train_losses, val_losses, train_accs, val_accs)
    
#     plot_predictions_and_targets(model, train_loader, "Train loader", 100)
#     plot_predictions_and_targets(model, val_loader, "Validation loader", 100)
#     plot_predictions_and_targets(model, test_loader, "Test loader", 100)
    
    
#     # Sauvegarder la configuration du modèle
#     config = {
#         'input_shape': input_size,
#         'output_shape': output_size,
#         'activation': 'ReLU',  # Sauvegarder le nom de l'activation
#         'n_layers' : n_layers,
#         'n_nodes' : n_nodes,
#         'L1_penalty': L1_penalty,
#         'L2_penalty': L2_penalty,
#         'use_batch_norm': use_batch_norm,
#         'dropout_prob': output_size
#     }

#     with open('model_config.json', 'w') as f:
#         json.dump(config, f)
#     torch.save(model.state_dict(), file_path)

def train_model_supervised_learning(params, criterion_class, criterion_params, train_loader, val_loader, test_loader, file_path) : 

    model, optimizer, criterion, scheduler, early_stopping = configure_parametres(params, criterion_class, criterion_params)

    # Entraînement du modèle
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(params['n_epochs']):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion)
        val_loss, val_acc = evaluate(model, val_loader, criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f'Epoch [{epoch+1}/{params['n_epochs']}], Train Loss: {train_losses[-1]:.6f}, Val Loss: {val_losses[-1]:.6f}, Train Acc: {train_acc:.6f}, Val Acc: {val_acc:.6f}')

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
    
    save_model(params, model, file_path)
    
    
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

# En chantier ...
# ---------------
# def find_best_hyperparametres(filename, retrain, file_path) : 
#     batch_size = 32
#     train_loader, val_loader, test_loader, input_size, output_size = prepare_data(batch_size, filename)
    
#     if retrain or os.path.exists(file_path) == False: 
        
#         # Définir le modèle
#         # input_size = 10
#         # output_size = 1
#         n_layers_list = [1]
#         n_nodes_list = [[12]]
#         activations_list = [[nn.GELU()]]
#         L1_penalty_list = [0.001, 0.01]
#         L2_penalty_list = [0.001, 0.01]
#         learning_rate_list = [1e-3]
#         criterion_list = [
#             (LogCoshLoss, {'factor': [1.0, 1.5, 1.8]}),
#             (ModifiedHuberLoss, {'delta': [0.2, 0.5, 1.0, 1.5, 2.0], 'factor': [1.0, 1.5, 2, 2.5, 3.0]}),
#             (ExponentialLoss, {'alpha': [0.5, 0.8, 1.0]})
#         ]
        
#         n_layers = 1
#         n_nodes = [10]
        
#         activation = nn.ReLU()
#         L1_penalty = 0.001
#         L2_penalty = 0.001
#         use_batch_norm = True
#         dropout_prob = 0.5
#         learning_rate = 1e-4

#         n_epochs = 1000
        
#         # Grid Search
#         best_val_loss = float('inf')
#         best_params = None
#         best_criterion_class = None
#         best_criterion_params = None
        
#         for params in product(n_layers_list, n_nodes_list, activations_list, L1_penalty_list, L2_penalty_list, learning_rate_list):
#             param_dict = {
#                 'n_layers': params[0],
#                 'n_nodes': params[1],
#                 'activations': params[2],
#                 'L1_penalty': params[3],
#                 'L2_penalty': params[4],
#                 'learning_rate': params[5]
#             }
#             for criterion_class, criterion_param_grid in criterion_list:
#                 for criterion_params_comb in product(*criterion_param_grid.values()):
#                     criterion_params = dict(zip(criterion_param_grid.keys(), criterion_params_comb))
#                     print(f'Training with parameters: {param_dict} and criterion: {criterion_class.__name__} with params: {criterion_params}')
#                     val_loss, val_acc = train_and_evaluate(param_dict, criterion_class, criterion_params)
#                     if val_loss < best_val_loss:
#                         best_val_loss = val_loss
#                         best_params = param_dict
#                         best_criterion_class = criterion_class
#                         best_criterion_params = criterion_params
#                     list_simulation.append(f"val_loss = {val_loss} and val acc = {val_acc} - Training with parameters: {param_dict} and criterion: {criterion_class.__name__} with params: {criterion_params}")

#         print(f'Best parameters found: {best_params} with validation loss: {best_val_loss}')
#         print(f'Best criterion: {best_criterion_class.__name__} with parameters: {best_criterion_params}')
        
#         train_model_supervised_learning(params, criterion_class, criterion_params, train_loader, val_loader, test_loader, file_path)
        
#     visualize_prediction(train_loader, val_loader, test_loader, file_path)
    

def configure_parametres(params, criterion_class, criterion_params):
    model = Model(params['input_size'], params['output_size'], params['n_layers'], params['n_nodes'], params['activations'],
                  params['L1_penalty'], params['L2_penalty'], True, 0.5)
    optimizer = torch.optim.Adam(model.parameters(), params['learning_rate'])
    criterion = criterion_class(**criterion_params)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=20, min_lr=1e-8, verbose=True)
    early_stopping = EarlyStopping(monitor='val_mae', patience=50, min_delta=0.00001, verbose=True)
    
    return model, optimizer, criterion, scheduler, early_stopping

