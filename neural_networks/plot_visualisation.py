from math import ceil
import matplotlib as plt
import matplotlib.pyplot as plt
import torch
import numpy as np
import os
from neural_networks.data_preparation import create_data_loader, get_y_and_labels
from neural_networks.file_directory_operations import create_and_save_plot
from neural_networks.other import compute_row_col
from neural_networks.Mode import Mode
from neural_networks.save_model import load_saved_model
import pandas as pd

def mean_distance(predictions, targets):
    """
    Compute mean distance beetween predictions and targets

    INPUTS :
    - predictions (torch.Tensor): Model's predictions 
    - targets (torch.Tensor): Targets

    OUTPUT : 
        float: mean distance
    """
    distance = torch.mean(torch.abs(predictions - targets))
    return distance.item()

def compute_pourcentage_error(predictions, targets) : 
    """
    Compute mean relative error beetween predictions and targets

    INPUTS :
    - predictions (torch.Tensor): Model's predictions 
    - targets (torch.Tensor): Targets

    OUTPUTS : 
        error_pourcentage : float, relative error
        error_pourcentage_abs : float, relative error in abs
    """
    error_pourcentage = torch.mean((torch.abs(predictions - targets)) / targets) * 100
    return error_pourcentage.item(), torch.abs(error_pourcentage).item()

def plot_loss_and_accuracy(train_losses, val_losses, train_accs, val_accs, file_path, show_plot = False):
    """Plot loss and accuracy (train and validation)

    INPUT :
    - train_losses : [float], all values of train loss, variation during trainning
    - val_losses : [float], all values of validation loss, variation during trainning
    - train_accs : [float], all values of train accuracy (mean distance), variation during trainning
    - val_accs : [float], all values of validation accuracy (mean distance), variation during trainning
    """
    
    # Create subplots
    _, axs = plt.subplots(1, 2, figsize=(15, 5))

    # Plot loss graph
    axs[0].plot(train_losses, label='Train Loss')
    axs[0].plot(val_losses, label='Validation Loss')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].set_title('Train and Validation Loss over Epochs')
    axs[0].legend()

    # Plot accuracy graph
    axs[1].plot(train_accs, label='Train Accuracy')
    axs[1].plot(val_accs, label='Validation Accuracy')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy')
    axs[1].set_title('Train and Validation Accuracy over Epochs')
    axs[1].legend()

    plt.tight_layout()
    
    create_and_save_plot(file_path, "plot_loss_and_accuracy")
    if show_plot == False : 
        plt.close()

# -----------------------------
def get_predictions_and_targets(model, data_loader, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    
    """Get predictions and targets from a model and data loader.

    INPUTS:
    - model:The trained PyTorch model to be evaluated.
    - data_loader: DataLoader containing the dataset to evaluate.
    - device: The device to run the model on (default is CUDA if available, otherwise CPU).

    OUTPUTS:
    - predictions: A list of predictions made by the model.
    - targets: A list of true target values from the dataset.
    """
    
    model.eval() # model in evaluation mode
    predictions = []
    targets = []
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            outputs = outputs.squeeze()  # Remove dimensions of size 1
            predictions.extend(outputs.cpu().numpy())
            targets.extend(labels.cpu().numpy())
    return predictions, targets
    
def plot_predictions_and_targets(model, y_labels, loader, string_loader, num, directory_path) :
    
    """Plot the true values and predicted values for a given model and data loader.
    
    INPUTS:
    - model: The trained PyTorch model to be evaluated.
    - y_labels : [string], all y (outputs of model) names columns 
    - loader: DataLoader containing the dataset to evaluate.
    - string_loader : string, loader name
    - num: The number of samples to plot for comparison.
    - directory_path : string, path to save plot

    OUTPUT:
    - None: The function generates a plot showing the true values and predicted values.
    """
    num_rows, num_cols = compute_row_col(len(y_labels), 3)
    predictions, targets = get_predictions_and_targets(model, loader)
    
    # special case if nbQ == 1
    if num_cols == 1 and num_rows == 1 : 
        acc = mean_distance(torch.tensor(np.array(predictions)), torch.tensor(np.array(targets)))
        error_pourcen, error_pourcen_abs = compute_pourcentage_error(torch.tensor(np.array(predictions)), torch.tensor(np.array(targets)))
        
        plt.figure(figsize=(10, 5))
        plt.plot(targets[:num], label='True values', marker='o')
        plt.plot(predictions[:num], label='Predictions', marker='o',linestyle='--')
        plt.xlabel('Sample')
        plt.ylabel(f"{y_labels[0]}")
        plt.title(f"Predictions and targets - {string_loader}, acc = {acc:.6f}, error% = {error_pourcen:.3f}%, error abs% = {error_pourcen_abs:.3f}%", fontweight='bold')
        plt.legend()
    
    else :  
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 10))
        axs = axs.flatten() if num_rows == 1 or num_cols == 1 else axs

        for k in range(len(y_labels)) :
            row = k // num_cols
            col = k % num_cols
            index = k if num_rows == 1 or num_cols == 1 else (row, col)
            
            acc = mean_distance(torch.tensor([prediction[k] for prediction in predictions]), torch.tensor([target[k] for target in targets]))
            error_pourcen, error_pourcen_abs = compute_pourcentage_error(torch.tensor([prediction[k] for prediction in predictions]), torch.tensor([target[k] for target in targets]))
        
            axs[index].plot([target[k] for target in targets][:num], label='True values', marker='^', markersize=2)
            axs[index].plot([prediction[k] for prediction in predictions][:num], label='Predictions', marker='o',linestyle='--', markersize=2)
            axs[index].set_xlabel("Sample")
            axs[index].set_ylabel("Value")
            axs[index].set_title(f"{y_labels[k]}, acc = {acc:.6f}, error% = {error_pourcen:.3f}%, error abs% = {error_pourcen_abs:.3f}%",fontsize='smaller')
            axs[index].legend()
        
        fig.suptitle(f"Predictions and targets - {string_loader}", fontweight='bold')
        plt.tight_layout()  
        
    create_and_save_plot(f"{directory_path}", f"plot_predictions_and_targets_{string_loader}.png")
    plt.show()

# ------------------------------------------
def get_predictions_and_targets_from_selected_y_labels(model, loader, y_labels, y_selected) :
    """Plot the true values and predicted values for a given model and data loader
    return only specifics columms of y_selected
    
    INPUTS:
    - model: The trained PyTorch model to be evaluated.
    - loader: DataLoader containing the dataset to evaluate.
    - y_labels : [string], all y (outputs of model) names columns 
    - y_selected: [string], all y (outputs of model) names columns selected 

    OUTPUTS:
    - selected_predictions : only predictions from columns y selected
    - selected_targets : only targets from columns y selected
    """
    
    predictions, targets = get_predictions_and_targets(model, loader)

    for y in y_selected : 
        if y not in y_labels : 
            TypeError("error : y isn't in y_labels") 
    
    if y_labels == y_selected : 
        return predictions, targets
            
    else : 
        # Keep only indices of y selected
        selected_indices = [y_labels.index(label) for label in y_selected]
        
        # selected only columns of y selected
        selected_predictions = [[row[i] for i in selected_indices] for row in predictions]
        selected_targets = [[row[i] for i in selected_indices] for row in targets]
        
        return selected_predictions, selected_targets

def general_plot_predictions(mode, mode_selected, folder_name, nbQ) :
    """Before all 'plot predictions', some preparations are necessary

    INPUTS : 
    - mode : Mode, mode of model
    - mode_selected : Mode, mode selected. It is always a mode equal are inferior to 'mode'
        Examples : mode_selected = mode or mode = Mode.LMT_DLMT_DQ and mode_selected = MUSCLE
    - folder_name : string, path to folder with all excel files q variation
    - nbQ : int, number of q in biorbd model

    OUTPUTS :
    - filenames : [string], name of each excel file q all variation
    - loaders : DataLoader containing the dataset to evaluate
    - row_fixed : int, number of row for subplot
    - col_fixed : int, number of col for subplot
    - y_labels : [string], all y (outputs of model) names columns 
    - y_selected: [string], all y (outputs of model) names columns selected 
    """
    
    # Get all filename, name off excel files
    all_possible_categories = [0,1,2,3,4,5,6,7,8,9,10,11]
    filenames = sorted([filename for filename in os.listdir(folder_name)])
    
    # Create loader and y_label
    loaders = []
    for filename in filenames[:nbQ]:
        loader, y_labels = create_data_loader(mode, f"{folder_name}/{filename}", all_possible_categories)
        loaders.append(loader)
        
    # Compute number of rows and cols for subplot    
    row_fixed, col_fixed = compute_row_col(nbQ, 3)
    
    # Get y_selected from mode_selected
    df_datas = pd.read_excel(f"{folder_name}/{filenames[0]}", nrows=0)
    _, y_selected = get_y_and_labels(mode_selected, df_datas, False)
    
    return filenames, loaders, row_fixed, col_fixed, y_labels, y_selected

def plot_predictions_and_targets_from_filenames(mode, mode_selected, model, nbQ, file_path, folder_name, num):
    """ Create plot to compare predictions and targets for ONE specific columns y (example : lmt, torque, ...)

    INPUTS : 
    - mode : Mode, mode of model
    - mode_selected : Mode, mode selected. It is always a mode equal are inferior to 'mode'
        Examples : mode_selected = mode or mode = Mode.LMT_DLMT_DQ and mode_selected = MUSCLE
    - model, pytorch model
    - nbQ : number of q in biorbd model
    - file_path : string, path to save the plot
    - folder_name : folder where excel files q all variaton are saved
    - num : int, number of points for the plot

    """
    filenames, loaders, row_fixed, col_fixed, y_labels, y_selected = general_plot_predictions(mode, mode_selected, folder_name, nbQ)
    fig, axs = plt.subplots(row_fixed,col_fixed, figsize=(15, 10))
    
    for q_index in range(nbQ) : 
        
        row = q_index // 3
        col = q_index % 3
        
        # Get selected predictions and targets from y_selected
        predictions, targets = get_predictions_and_targets_from_selected_y_labels(model, loaders[q_index], y_labels, y_selected)
        # Compute accuracy
        acc = mean_distance(torch.tensor(np.array(predictions)), torch.tensor(np.array(targets)))
        error_pourcen, error_pourcen_abs = compute_pourcentage_error(torch.tensor(np.array(predictions)), torch.tensor(np.array(targets)))

        axs[row, col].plot(targets[:num], label='True values', marker='o', markersize=2)
        axs[row, col].plot(predictions[:num], label='Predictions', marker='D', linestyle='--', markersize=2)
        axs[row, col].set_title(f"File: {filenames[q_index].replace(".xlsx", "")},\n acc = {acc:.6f}, error% = {error_pourcen:.3f}%, error abs% = {error_pourcen_abs:.3f}%",fontsize='smaller')
        axs[row, col].set_xlabel(f'q{q_index} Variation',fontsize='smaller')
        axs[row, col].set_ylabel(f'{y_selected[0]}',fontsize='smaller')
        axs[row, col].legend()
    
    fig.suptitle(f'Predictions and targets of {y_selected[0]}', fontweight='bold')
    plt.tight_layout()  
    create_and_save_plot(f"{file_path}", f"plot_{y_selected[0]}_predictions_and_targets.png")
    plt.show()
        
def plot_predictions_and_targets_from_filenames_dlmt_dq(mode, mode_selected, model, nbQ, file_path, folder_name, num):
    """ Create plot to compare predictions and targets for DLMT_DQ

    INPUTS : 
    - mode : Mode, mode of model
    - mode_selected : Mode, mode selected. It is always a mode equal are inferior to 'mode'
        Examples : mode_selected = mode or mode = Mode.LMT_DLMT_DQ and mode_selected = MUSCLE
    - model, pytorch model
    - nbQ : number of q in biorbd model
    - file_path : string, path to save the plot
    - folder_name : folder where excel files q all variaton are saved
    - num : int, number of points for the plot

    """
    
    filenames, loaders, row_fixed, col_fixed, y_labels, y_selected = general_plot_predictions(mode, mode_selected, folder_name, nbQ)
    
    # For each q index, create a figure
    for q_index in range(nbQ) : 
        fig, axs = plt.subplots(row_fixed, col_fixed, figsize=(15, 10))
        
        # Get selected predictions and targets from y_selected, one excel file
        predictions, targets = get_predictions_and_targets_from_selected_y_labels(model, loaders[q_index], y_labels, y_selected)
        
        # Special case if nbQ == 1
        if row_fixed == 1 and col_fixed == 1 : 
            acc = mean_distance(torch.tensor([prediction[0] for prediction in predictions]), torch.tensor([target[0] for target in targets]))
            error_pourcen, error_pourcen_abs = compute_pourcentage_error(torch.tensor([prediction[0] for prediction in predictions]), torch.tensor([target[0] for target in targets]))
            
            plt.figure(figsize=(10, 5))
            plt.plot(targets[:num], label='True values', marker='o')
            plt.plot(predictions[:num], label='Predictions', marker='o',linestyle='--')
            plt.xlabel('q variation')
            plt.ylabel(f"{y_selected[0]}")
            plt.title(f"Predictions and targets of Lever Arm, q{q_index} variation, acc = {acc:.6f}, error% = {error_pourcen:.3f}%, error abs% = {error_pourcen_abs:.3f}%", fontweight='bold')
            plt.legend()
        
        else : 
            # else, subplot of each q
            for i in range (len(y_selected)) : 
                acc = mean_distance(torch.tensor([prediction[i] for prediction in predictions]), torch.tensor([target[i] for target in targets]))
                error_pourcen, error_pourcen_abs = compute_pourcentage_error(torch.tensor([prediction[i] for prediction in predictions]), torch.tensor([target[i] for target in targets]))
                
                row = i // 3
                col = i % 3
            
                axs[row, col].plot([target[i] for target in targets][:num], label='True values', marker='o', markersize=2)
                axs[row, col].plot([prediction[i] for prediction in predictions][:num], label='Predictions', marker='D', linestyle='--', markersize=2)
                axs[row, col].set_title(f"File: {filenames[q_index].replace(".xlsx", "")},\n acc = {acc:.6f}, error% = {error_pourcen:.3f}%, error abs% = {error_pourcen_abs:.3f}%",fontsize='smaller')
                axs[row, col].set_xlabel(f'q{q_index} Variation',fontsize='smaller')
                axs[row, col].set_ylabel(f'dlmt_dq{i}',fontsize='smaller')
                axs[row, col].legend()
        
            fig.suptitle(f'Predictions and targets of Lever Arm, q{q_index} variation', fontweight='bold')
            plt.tight_layout()  
            create_and_save_plot(f"{file_path}", f"q{q_index}_plot_length_jacobian_predictions_and_targets.png")
            plt.show()
    
    return None
# -------------------------------------
def visualize_prediction_trainning(model, file_path, y_labels, train_loader, val_loader, test_loader) : 
    """Create plots 'predictions and targets' for train, validation and test loader

    INPUTS :
    - model : pytorch model
    - file_path : string, path to save plot
    - y_labels : [string] all name of columns y (output of model)
    - train_loader : DataLoader, data trainning (80% of 80%)
    - val_loader : DataLoader, data validation (20% of 80%)
    - test_loader : DataLoader, data testing (20%)
    """
    
    plot_predictions_and_targets(model, y_labels, train_loader, "Train loader", 100, file_path)
    plot_predictions_and_targets(model, y_labels, val_loader, "Validation loader", 100, file_path)
    plot_predictions_and_targets(model, y_labels , test_loader, "Test loader", 100, file_path)

def visualize_prediction(mode, nbQ, file_path, folder_name_for_prediction) : 
    
    """
    Load saved model and plot-save visualisations 
    
    INPUTS 
    - mode : Mode
    - nbQ : int, number of q in biorbd model
    - file_path : string, path where the file 'model_config.json' of the model could be find
    - folder_name_for_prediction : string, path where files of folder 'plot_all_q_variation_' could be find
    """

    model = load_saved_model(file_path)
    
    if mode == Mode.DLMT_DQ : 
        plot_predictions_and_targets_from_filenames_dlmt_dq(mode, mode, model, nbQ, file_path, folder_name_for_prediction, 100)
    elif mode == Mode.MUSCLE_DLMT_DQ : 
        plot_predictions_and_targets_from_filenames(mode, Mode.MUSCLE, model, nbQ, file_path, folder_name_for_prediction, 100)
        plot_predictions_and_targets_from_filenames_dlmt_dq(mode, Mode.DLMT_DQ, model, nbQ, file_path, folder_name_for_prediction, 100)
    elif mode == Mode.TORQUE_MUS_DLMT_DQ : 
        plot_predictions_and_targets_from_filenames(mode, Mode.MUSCLE, model, nbQ, file_path, folder_name_for_prediction, 100)
        plot_predictions_and_targets_from_filenames_dlmt_dq(mode, Mode.DLMT_DQ, model, nbQ, file_path, folder_name_for_prediction, 100)
        plot_predictions_and_targets_from_filenames(mode, Mode.TORQUE, model, nbQ, file_path, folder_name_for_prediction, 100)
    else : # MUSCLE or TORQUE
        plot_predictions_and_targets_from_filenames(mode, mode, model, nbQ, file_path, folder_name_for_prediction, 100)


    
