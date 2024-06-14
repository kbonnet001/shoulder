import torch
import torch.nn as nn
from neural_networks.Model import Model
from neural_networks.plot_visualisation import plot_predictions_and_targets
import json

def save_model(params, model, file_path) : 
    # Sauvegarder la configuration du modèle
    config = {
        'input_shape': params['input_size'],
        'output_shape': params['output_size'],
        'activation': 'ReLU',  # Sauvegarder le nom de l'activation
        'n_layers' : params['n_layers'],
        'n_nodes' : params['n_nodes'],
        'L1_penalty': params['L1_penalty'],
        'L2_penalty': params['L2_penalty'],
        'use_batch_norm': params['use_batch_norm'],
        'dropout_prob': params['dropout_prob']
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