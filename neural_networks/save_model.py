import torch
import torch.nn as nn
from neural_networks.Model import Model
from neural_networks.plot_visualisation import plot_predictions_and_targets
from neural_networks.file_directory_operations import create_and_save_plot
from neural_networks.ModelHyperparameters import ModelHyperparameters
import json
   
    
def save_model(model, input_size, output_size, Hyperparams, file_path) : 
    """Save a model with its parameters and hyperparameters
    
    INPUTS : 
    - model : model to save
    - input_size : int, size of input X
    - output_size : int, size of output y
    - Hyperparams : ModelHyperparameters, all hyperparameter 
    - file_path : string, path """
    
    # Save model configuation
    config = {
        'input_size': input_size,
        'output_size': output_size,
        'activation': Hyperparams.activation_names, 
        'n_layers': Hyperparams.n_layers,
        'n_nodes': Hyperparams.n_nodes,
        'L1_penalty': Hyperparams.L1_penalty,
        'L2_penalty': Hyperparams.L2_penalty,
        'use_batch_norm': Hyperparams.use_batch_norm,
        'dropout_prob': Hyperparams.dropout_prob
    }

    with open('model_config.json', 'w') as f:
        json.dump(config, f)
    torch.save(model.state_dict(), file_path)
    
def visualize_prediction(train_loader, val_loader, test_loader, file_path) : 
    
    """Load saved model and plot-save visualisation 
    
    INPUTS 
    """
    
    # Charger la configuration du modèle
    with open('model_config.json', 'r') as f:
        config = json.load(f)

    # Recréer l'activation
    activations = [getattr(nn, activation)() for activation in config['activation']]

    # Recréer le modèle avec la configuration chargée
    model = Model(
        input_size=config['input_size'],
        output_size=config['output_size'],
        activations=activations,
        n_layers=config['n_layers'], 
        n_nodes=config['n_nodes'],
        L1_penalty=config['L1_penalty'],
        L2_penalty=config['L2_penalty'],
        use_batch_norm=config['use_batch_norm'],
        dropout_prob=config['dropout_prob']
    )

    model.load_state_dict(torch.load(f"{file_path}/model"))
    model.eval()
    
    plot_predictions_and_targets(model, train_loader, "Train loader", 100, file_path, "train_loader")
    plot_predictions_and_targets(model, val_loader, "Validation loader", 100, file_path, "val_loader")
    plot_predictions_and_targets(model, test_loader, "Test loader", 100, file_path, "test_loader")