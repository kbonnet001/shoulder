import torch
import torch.nn as nn
from neural_networks.Model import Model
import json
from neural_networks.Timer import measure_time    
import os

def del_saved_model(file_path) : 
    
    # Paths for model and config files
    model_path = os.path.join(file_path, "model")
    config_path = os.path.join(file_path, "model_config.json")
    
    # Remove existing files if they exist
    if os.path.exists(model_path):
        os.remove(model_path)
    if os.path.exists(config_path):
        os.remove(config_path)

def save_model(model, input_size, output_size, Hyperparams, file_path): 
    """
    Save a model with its parameters and hyperparameters
    
    INPUTS: 
    - model: model to save
    - input_size: int, size of input X
    - output_size: int, size of output y
    - Hyperparams: ModelHyperparameters, all hyperparameter 
    - file_path: string, path 
    """
    
    # Paths for model and config files
    model_path = os.path.join(file_path, "model")
    config_path = os.path.join(file_path, "model_config.json")
    
    del_saved_model(file_path)

    # Save model configuration
    config = {
        'input_size': input_size,
        'output_size': output_size,
        'activation': Hyperparams.activation_names, 
        'n_nodes': Hyperparams.n_nodes,
        'L1_penalty': Hyperparams.L1_penalty,
        'L2_penalty': Hyperparams.L2_penalty,
        'use_batch_norm': Hyperparams.use_batch_norm,
        'dropout_prob': Hyperparams.dropout_prob
    }

    with open(config_path, 'w') as f:
        json.dump(config, f)
    torch.save(model.state_dict(), model_path)


def load_saved_model(file_path) : 
    
    """
    Load a saved model from file_path
    
    INPUT : 
    - file_path : string, path where the file 'model_config.json' of the model could be find
    
    OUTPUT : 
    - model : model loaded in eval mode
    """
    
    # Charger la configuration du modèle
    with open(f'{file_path.replace("/model", "")}/model_config.json', 'r') as f:
        config = json.load(f)

    # Recréer l'activation
    activations = [getattr(nn, activation)() for activation in config['activation']]

    # Recréer le modèle avec la configuration chargée
    model = Model(
        input_size=config['input_size'],
        output_size=config['output_size'],
        activations=activations,
        n_nodes=config['n_nodes'],
        L1_penalty=config['L1_penalty'],
        L2_penalty=config['L2_penalty'],
        use_batch_norm=config['use_batch_norm'],
        dropout_prob=config['dropout_prob']
    )

    model.load_state_dict(torch.load(f"{file_path}/model"))
    model.eval()

    return model

def main_function_model(file_path, inputs) : 
    """
    Model prediction with a saved model
    Please, look at this function as an example 
    One load before all is beter than one load for each time you use the model prediction ... 

    INPUTS : 
    - file_path : string, path where the file 'model_config.json' of the model could be find
    - inputs : [], inputs, check is the dimention is correct before
    
    OUTPUT :
    - output : pytorch tensor, model's prediction(s) 
    """
    # You must have a torch tensor !
    if not isinstance(inputs, torch.Tensor):
        inputs = torch.tensor([inputs])
    
    if len(inputs.size()) == 1 : # security when y is 1
        inputs = inputs.unsqueeze(0)
        
    # Load model from file_path, model eval    
    with measure_time() as model_load_timer:
      model = load_saved_model(file_path)
    
    with measure_time() as model_timer:
        outputs = model(inputs).squeeze()
    
    # print(f"output(s) = {outputs}, time execution (without loading model time) = {model_timer.execution_time}")
    return outputs, model_load_timer, model_timer
