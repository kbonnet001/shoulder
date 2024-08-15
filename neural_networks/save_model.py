import torch
import torch.nn as nn
from neural_networks.Model import Model
import json
from neural_networks.Timer import measure_time    
import os

def del_saved_model(file_path) : 
    """ Deletes the saved model and its configuration file from the specified directory.

    Args:
    - file_path (str): The directory path where the model and configuration files are stored.
    """
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
    Saves a model along with its configuration and hyperparameters to the specified directory.

    Args:
    - model: The PyTorch model to save.
    - input_size (int): The size of the input layer.
    - output_size (int): The size of the output layer.
    - Hyperparams (ModelHyperparameters): An obj containing the model's hyperparameters.
    - file_path (str): The directory path where the model and configuration will be saved.
    """
    
    # Paths for model and config files
    model_path = os.path.join(file_path, "model")
    config_path = os.path.join(file_path, "model_config.json")
    
    # Delete any existing saved model and configuration to avoid conflicts
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
    
    # Save the configuration as a JSON file
    with open(config_path, 'w') as f:
        json.dump(config, f)
    # Save the model's state dictionary
    torch.save(model.state_dict(), model_path)

def load_saved_model(file_path) : 
    
    """
    Loads a saved model from the specified file path.

    Args:
    - file_path (str): Path to the directory containing 'model_config.json'.

    Returns:
    - model: The loaded model in evaluation mode.
    """
    
    # Load the model configuration from the JSON file
    with open(f'{file_path.replace("/model", "")}/model_config.json', 'r') as f:
        config = json.load(f)

    # Recreate the activation functions based on the loaded configuration
    activations = [getattr(nn, activation)() for activation in config['activation']]

    # Recreate the model using the loaded configuration
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

    # Load the saved model state dictionary
    model.load_state_dict(torch.load(f"{file_path}/model"))

    # Set the model to evaluation mode
    model.eval()

    return model

def main_function_model(file_path, inputs) : 
    """
    Perform model prediction using a saved model.

    This function demonstrates loading a model once and using it multiple times for predictions,
    which is more efficient than loading the model each time a prediction is needed.

    Args:
    - file_path (str): Path to the directory containing 'model_config.json'.
    - inputs (list or torch.Tensor): The inputs for the model. Ensure the dimensions are correct.

    Returns:
    - outputs (torch.Tensor): The model's prediction(s).
    - model_load_timer: The time taken to load the model.
    - model_timer: The time taken to make predictions.
    """
    # Ensure the inputs are a PyTorch tensor
    if not isinstance(inputs, torch.Tensor):
        inputs = torch.tensor([inputs])
    
    if len(inputs.size()) == 1 : # security when y is 1
        inputs = inputs.unsqueeze(0)
        
    # Load model from file_path, model eval    
    with measure_time() as model_load_timer:
      model = load_saved_model(file_path)
      
    # Perform model prediction and measure execution time
    with measure_time() as model_timer:
        outputs = model(inputs).squeeze()
    
    # Uncomment the line below to print the outputs and execution time
    # print(f"output(s) = {outputs}, time execution (without loading model time) = {model_timer.execution_time}")
    
    return outputs, model_load_timer, model_timer

