import torch
import torch.nn as nn
from neural_networks.Model import Model
import json
from neural_networks.Timer import measure_time
   
    
def save_model(model, input_size, output_size, Hyperparams, file_path) : 
    """
    Save a model with its parameters and hyperparameters
    
    INPUTS : 
    - model : model to save
    - input_size : int, size of input X
    - output_size : int, size of output y
    - Hyperparams : ModelHyperparameters, all hyperparameter 
    - file_path : string, path 
    """
    
    # Save model configuation
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

    with open(f'{file_path}/model_config.json', 'w') as f:
        json.dump(config, f)
    torch.save(model.state_dict(), f"{file_path}/model")

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
    
    # Load model from file_path, model eval
    model = load_saved_model(file_path)
    
    with measure_time() as timer:
        # You must have a torch tensor !
        inputs_tensor = torch.tensor([inputs])
        outputs = model(inputs_tensor).squeeze()
    
    print(f"output(s) = {outputs}, time execution (without loading model time) = {timer.execution_time}")
    return outputs
