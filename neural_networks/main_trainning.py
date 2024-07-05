from neural_networks.data_preparation import data_preparation_create_tensor, compute_samples, create_loaders_from_folder
from neural_networks.data_tranning import train, evaluate
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
from itertools import product
from neural_networks.Loss import *
from neural_networks.ModelHyperparameters import ModelHyperparameters
from neural_networks.jesaispas import create_directory, create_and_save_plot
import time

def train_model_supervised_learning(train_loader, val_loader, test_loader, input_size, output_size, Hyperparams, 
                                                                                                            file_path, plot = False, save = False) : 
    model = Model(input_size, output_size, Hyperparams.n_layers, Hyperparams.n_nodes, Hyperparams.activations, 
                  Hyperparams.L1_penalty, Hyperparams.L2_penalty, Hyperparams.use_batch_norm, Hyperparams.dropout_prob)
    
    Hyperparams.compute_optimiser(model)
    
    if plot : 
        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []

    # Initialization of ReduceLROnPlateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(Hyperparams.optimizer, mode='min', factor=0.1, patience=20, min_lr=1e-8, verbose=True)
    # Initialization of EarlyStopping
    early_stopping = EarlyStopping(monitor='val_mae', patience=50, min_delta=0.00001, verbose=True)

    for epoch in range(Hyperparams.num_epochs):
        train_loss, train_acc = train(model, train_loader, Hyperparams.optimizer, Hyperparams.criterion)
        val_loss, val_acc = evaluate(model, val_loader, Hyperparams.criterion)
        
        if plot : 
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)

        print(f'Epoch [{epoch+1}/{Hyperparams.num_epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Train Acc: {train_acc:.6f}, Val Acc: {val_acc:.6f}')

        # Réduire le taux d'apprentissage si nécessaire
        scheduler.step(val_loss)

        # Vérifier l'arrêt précoce
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping at epoch:", epoch+1)
            break

    # Évaluation du modèle sur l'ensemble de test
    test_loss, test_acc = evaluate(model, test_loader, Hyperparams.criterion)
    print(f'Test Loss: {test_loss:.6f}, Test Acc: {test_acc:.6f}')
    
    if plot : 
        plot_loss_and_accuracy(train_losses, val_losses, train_accs, val_accs)
        create_and_save_plot(Hyperparams.model_name, "plot_loss_and_accuracy")
        
        plot_predictions_and_targets(model, train_loader, "Train loader", 100)
        create_and_save_plot(Hyperparams.model_name, "plot_predictions_and_targets_train_loader")
        plot_predictions_and_targets(model, val_loader, "Validation loader", 100)
        create_and_save_plot(Hyperparams.model_name, "plot_predictions_and_targets_val_loader")
        plot_predictions_and_targets(model, test_loader, "Test loader", 100)
        create_and_save_plot(Hyperparams.model_name, "plot_predictions_and_targets_test_loader")
    
    # Save model
    if save : 
        save_model(model, input_size, output_size, Hyperparams, file_path)
    
    return val_loss, val_acc
    
def main_superised_learning(Hyperparams, folder_name, retrain, file_path, plot_preparation) : 
    
    """Main fonction for prepare, train-val-test and save a model 
    
    - Hyperparams : (ModelHyperparameters) all hyperparameters choosen by user
    - folder_name : string, path/name of the folder contained all excel data file of muscles (one for each muscle)
    - retrain : bool, True to train the model again
    - file_path : string, name of the model will be save after tranning
    - plot_preparation : bool, True to show distribution of all datas preparation"""
    
    # Create a folder for save plots
    create_directory(Hyperparams.model_name)

    train_loader, val_loader, test_loader, input_size, output_size = create_loaders_from_folder(Hyperparams, folder_name, plot_preparation)
    
    # train_model if retrain == True or if none file_path already exist
    if retrain or os.path.exists(file_path) == False: 
        
        train_model_supervised_learning(train_loader, val_loader, test_loader, input_size, output_size, Hyperparams, file_path)
        
    visualize_prediction(train_loader, val_loader, test_loader, file_path)
    
    
def find_best_hyperparameters(Hyperparams, folder_name) : 
    
    # Before beggining, compute an estimation of execution time
    # The user can choose to stop if the execution is to long according to him 
    # For example, if estimed execution time if around 100 hours... maybe you have to many hyperparameters to try ...
    total_time_estimated_s, total_time_estimated_min, total_time_estimated_h = compute_time_testing_hyperparams(
        Hyperparams, time_per_configuration_secondes = 60)
    
    print(f"------------------------\n"
          f"Time estimated for testing all configurations: \n- {total_time_estimated_s} secondes"
          f"\n- {total_time_estimated_min} minutes\n- {total_time_estimated_h} hours\n\n"
          f"Research of best hyperparameters will beggining in few secondes ...\n"
          f"------------------------")
    time.sleep(10) #10 
    
    print("Let's go !")
    
    ###
    # debut de la recherche

    train_loader, val_loader, test_loader, input_size, output_size = create_loaders_from_folder(Hyperparams, folder_name, plot = False)

    list_simulation= []

    # Grid Search
    best_val_loss = float('inf')
    best_params = None
    best_criterion_class = None
    best_criterion_params = None

    for params in product(Hyperparams.n_layers, Hyperparams.n_nodes, Hyperparams.activations, Hyperparams.L1_penalty
                          , Hyperparams.L2_penalty,Hyperparams.learning_rate, Hyperparams.dropout_prob):
        # param_dict = {
        #     'n_layers': params[0],
        #     'n_nodes': params[1],
        #     'activations': params[2],
        #     'L1_penalty': params[3],
        #     'L2_penalty': params[4],
        #     'learning_rate': params[5],
        #     'dropout_prob': params[6]
        # }
        
        try_hyperparams = ModelHyperparameters("Try Hyperparams", Hyperparams.batch_size, params[0], params[1], 
                                                params[2], "", params[3], params[4], 
                                                params[5], Hyperparams.num_epochs, None, 
                                                params[6], Hyperparams.use_batch_norm)
        
        for criterion_class, criterion_param_grid in Hyperparams.criterion:
            for criterion_params_comb in product(*criterion_param_grid.values()):
                criterion_params = dict(zip(criterion_param_grid.keys(), criterion_params_comb))
                try_hyperparams.add_criterion(criterion_class(**criterion_params))
                
                
                # print(f'Training with parameters: {param_dict} and criterion: {criterion_class.__name__} with params: {criterion_params}')
                print(try_hyperparams)
                # on fait l'entrainement
                val_loss, val_acc = train_model_supervised_learning(train_loader, val_loader, test_loader, input_size, output_size, try_hyperparams, 
                                                                                                            file_path="", plot = False)
                # on regarde si l'entrainement est bien ou pas
                if val_loss < best_val_loss:
                    # best_val_loss = val_loss
                    # best_params = param_dict
                    best_criterion_class = criterion_class
                    best_criterion_params = criterion_params
                    # Best hyperparameters
                    # best_criterion = best_criterion_class(**best_criterion_params)
                    best_hyperparameters = ModelHyperparameters("Best_hyperparameter", Hyperparams.batch_size, try_hyperparams.n_layers,
                                                try_hyperparams.n_nodes, try_hyperparams.activations, "", 
                                                try_hyperparams.L1_penalty, try_hyperparams.L2_penalty, 
                                                try_hyperparams.learning_rate, Hyperparams.num_epochs, try_hyperparams.criterion, 
                                                try_hyperparams.dropout_prob, Hyperparams.use_batch_norm)
                # dans tous les cas, on garde de cote dans une liste
                # list_simulation.append(f"val_loss = {val_loss} and val acc = {val_acc} - Training with parameters: {param_dict} and criterion: {criterion_class.__name__} with params: {criterion_params}")
                list_simulation.append(f"val_loss = {val_loss} and val acc = {val_acc} - Training with hyperparameters : {try_hyperparams} \ncriterion: {criterion_class.__name__} with parameters: {criterion_params}")


    # print(f'Best parameters found: {best_params} with validation loss: {best_val_loss}')
    # print(f'Best criterion: {best_criterion_class.__name__} with parameters: {best_criterion_params}')
    print(f"Best hyperparameters found : {best_hyperparameters}")
    print(f'Best criterion: {best_criterion_class.__name__} with parameters: {best_criterion_params}')

    # Best hyperparameters
    # best_criterion = best_criterion_class(**best_criterion_params)
    # best_hyperparameters = ModelHyperparameters("Best_hyperparameter", Hyperparams.batch_size, best_params['n_layers'],
    #                                             best_params['n_nodes'], best_params['activations'], "", 
    #                                             best_params['L1_penalty'], best_params['L2_penalty'], 
    #                                             best_params['learning_rate'], Hyperparams.num_epochs, best_criterion, 
    #                                             best_params['dropout_prob'], Hyperparams.use_batch_norm)
    
    return best_hyperparameters

#####

def compute_time_testing_hyperparams(Hyperparams, time_per_configuration_secondes = 60) : 
    
    #dans cette situation, hyperparams c'est des listes de trucs aue l'on veut tester
    n_combinations = (len(Hyperparams.n_layers) * len(Hyperparams.n_nodes) * len(Hyperparams.activations) *
                      len(Hyperparams.L1_penalty) * len(Hyperparams.L2_penalty) * len(Hyperparams.learning_rate) * 
                      len(Hyperparams.dropout_prob) * sum(len(vals) for _, vals in Hyperparams.criterion))

    total_time_estimated_secondes = n_combinations * time_per_configuration_secondes
    total_time_estimated_minutes = n_combinations * time_per_configuration_secondes / 60
    total_time_estimated_hours = n_combinations * time_per_configuration_secondes / 3600
    
    return total_time_estimated_secondes, total_time_estimated_minutes, total_time_estimated_hours
    
# input_size = 11
# output_size = 1
# n_layers_list = [1]
# n_nodes_list = [[12]]
# activations_list = [[nn.GELU(), nn.GELU()]]
# # L1_penalty_list = [0.001, 0.01]
# # L2_penalty_list = [0.001, 0.01]
# learning_rate_list = [1e-3]
# p_dropout = [0.2]

# criterion_list = [
#     (LogCoshLoss, {'factor': [1.0, 1.5, 1.8]}),
#     (ModifiedHuberLoss, {'delta': [0.2, 0.5, 1.0, 1.5, 2.0], 'factor': [1.0, 1.5, 2, 2.5, 3.0]}),
#     (ExponentialLoss, {'alpha': [0.5, 0.8, 1.0]})
# ]