import os
import matplotlib.pyplot as plt

def create_directory(directory_path):
    """
    Create a new folder

    Args :
        directory_path (str): path of the new folder
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"The folder'{directory_path}' have been created.")
    else:
        print(f"The folder '{directory_path}' already exist.")

def create_and_save_plot(directory_path, file_name):
    """
    Save a Matplotlib fig in a specific folder

    Args :
        directory_path (str): Path where the fig will be save.
        file_name (str): Name of the figure
    """
    
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    
    # Enregistre la figure
    file_path = os.path.join(directory_path, file_name)
    plt.savefig(file_path)

def save_text_to_file(text, file_path):
    """_summary_

    Args:
        text (_type_): _description_
        file_path (_type_): _description_
    """
    with open(file_path, 'w') as file:
        file.write(text)


def save_informations_model(file_path, num_try, val_loss, val_acc, train_timer, mean_model_load_timer, mean_model_timer, try_hyperparams, epoch, criterion_name, criterion_params) : 
    
    text = (
        f"num_try = {num_try}\n"
        f"val_loss = {val_loss}\n"
        f"val_acc = {val_acc}\n"
        f"execution_time_train = {train_timer}\n"
        f"execution_time_load_saved_model = {mean_model_load_timer}\n"
        f"execution_time_use_saved_model = {mean_model_timer}\n"
        f"mode = {try_hyperparams.mode}\n"
        f"batch_size = {try_hyperparams.batch_size}\n"
        f"n_nodes = {try_hyperparams.n_nodes}\n"
        f"activations = {try_hyperparams.activations}\n"
        f"L1_penalty = {try_hyperparams.L1_penalty}\n"
        f"L2_penalty = {try_hyperparams.L2_penalty}\n"
        f"learning_rate = {try_hyperparams.learning_rate}\n"
        # f"optimizer = {try_hyperparams.optimizer}\n"
        f"dropout_prob = {try_hyperparams.dropout_prob}\n"
        f"use_batch_norm = {try_hyperparams.use_batch_norm}\n"
        f"num_epochs_used = {epoch}\n"
        f"criterion_name = {criterion_name}\n"
        f"criterion_params = {criterion_params}"
    )
    
    save_text_to_file(text, f"{file_path}/model_informations.txt")
    
def read_info_model(file_path, infos):
    extracted_values = []
    
    with open(file_path, 'r') as file:
        for line in file:
            # Split the line into key and value
            parts = line.split(" = ")
            key, value = parts
            value = value.strip().strip('\n')
            
            # Check if the key is in the infos list
            if key in infos:
                # Try to convert value to a number, if possible
                try:
                    value = float(value)
                except ValueError:
                    pass
                extracted_values.append(value)
    
    return extracted_values
