from math import ceil
import matplotlib as plt
import matplotlib.pyplot as plt
import torch
import numpy as np
import os
from neural_networks.data_preparation import create_data_loader, get_y_and_labels
from neural_networks.file_directory_operations import create_and_save_plot, read_info_model
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

    OUPUT : 
        float: mean distance
    """
    distance = torch.mean(torch.abs(predictions - targets))
    return distance.item()

def compute_pourcentage_error(predictions, targets) : 
    """
    Compute mean distance beetween predictions and targets

    INPUTS :
    - predictions (torch.Tensor): Model's predictions 
    - targets (torch.Tensor): Targets

    OUPUT : 
        float: mean pourcentage of error predictions
    """
    # error_pourcentage = torch.mean((torch.abs(predictions - targets)) / targets) * 100
    error_pourcentage = torch.mean((torch.abs(predictions - targets))) / torch.mean(targets) * 100
    return torch.abs(error_pourcentage).item()

def plot_loss_and_accuracy(train_losses, val_losses, train_accs, val_accs, file_path):
    """Plot loss and accuracy (train and validation)

    INPUT :
    - train_losses :
    - val_losses
    - train_accs
    - val_accs """
    
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
    plt.show()

# -----------------------------
def get_predictions_and_targets(model, data_loader, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    
    """Get predictions and targets from a model and data loader.

    INPUT:
    - model: The trained PyTorch model to be evaluated.
    - data_loader: DataLoader containing the dataset to evaluate.
    - device: The device to run the model on (default is CUDA if available, otherwise CPU).

    OUTPUT:
    - predictions: A list of predictions made by the model.
    - targets: A list of true target values from the dataset.
    """
    
    model.eval()
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
    
def plot_predictions_and_targets(model, y_labels, loader, string_loader, num, directory_path, loader_name) :
    
    """Plot the true values and predicted values for a given model and data loader.
    INPUT:
    - model: The trained PyTorch model to be evaluated.
    - loader: DataLoader containing the dataset to evaluate.
    - num: The number of samples to plot for comparison.

    OUTPUT:
    - None: The function generates a plot showing the true values and predicted values.
    """
    num_rows, num_cols = compute_row_col(len(y_labels), 3)
    predictions, targets = get_predictions_and_targets(model, loader)
    
    if num_cols == 1 and num_rows == 1 : 
        acc = mean_distance(torch.tensor(np.array(predictions)), torch.tensor(np.array(targets)))
        error_pourcentage = compute_pourcentage_error(torch.tensor(np.array(predictions)), torch.tensor(np.array(targets)))
        
        plt.figure(figsize=(10, 5))
        plt.plot(targets[:num], label='True values', marker='o')
        plt.plot(predictions[:num], label='Predictions', marker='o',linestyle='--')
        plt.xlabel('Sample')
        plt.ylabel(f"{y_labels[0]}")
        plt.title(f"Predictions and targets - {string_loader}, acc = {acc:.6f}, error% = {error_pourcentage:.3f}%", fontweight='bold')
        plt.legend()
    
    else :  
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 10))
        axs = axs.flatten() if num_rows == 1 or num_cols == 1 else axs

        for k in range(len(y_labels)) :
            row = k // num_cols
            col = k % num_cols
            index = k if num_rows == 1 or num_cols == 1 else (row, col)
            
            acc = mean_distance(torch.tensor([prediction[k] for prediction in predictions]), torch.tensor([target[k] for target in targets]))
            error_pourcentage = compute_pourcentage_error(torch.tensor([prediction[k] for prediction in predictions]), torch.tensor([target[k] for target in targets]))
        
            axs[index].plot([target[k] for target in targets][:num], label='True values', marker='^', markersize=2)
            axs[index].plot([prediction[k] for prediction in predictions][:num], label='Predictions', marker='o',linestyle='--', markersize=2)
            axs[index].set_xlabel('Sample')
            axs[index].set_ylabel("Value")
            axs[index].set_title(f'{y_labels[k]}, acc = {acc:.6f}, error% = {error_pourcentage:.3f}%',fontsize='smaller')
            axs[index].legend()
        
        fig.suptitle(f"Predictions and targets - {string_loader}", fontweight='bold')
        plt.tight_layout()  
        
    create_and_save_plot(f"{directory_path}", f"plot_predictions_and_targets_{string_loader}.png")
    plt.show()

# ------------------------------------------
def get_predictions_and_targets_from_selected_y_labels(model, loader, y_labels, y_selected) :
    # on fait les predictions avec le mode reel 
    predictions, targets = get_predictions_and_targets(model, loader)

    if y_labels == y_selected : 
        return predictions, targets
    else : 
        # grace à y_labels et y_selected, on sait quelle colonne prendre
        # y_labels c'est la ref, donc synchro avec le mode reel
        selected_indices = [y_labels.index(label) for label in y_selected]
        
        selected_predictions = [[row[i] for i in selected_indices] for row in predictions]
        selected_targets = [[row[i] for i in selected_indices] for row in targets]
        
        return selected_predictions, selected_targets

def general_plot_predictions(mode, mode_selected, folder_name, nbQ) :
    # le truc general fait au debut de chaque function plot predictions and targets
    all_possible_categories = [0,1,2,3,4,5,6,7,8,9,10,11]
    filenames = sorted([filename for filename in os.listdir(folder_name)])
    
    loaders = []
    for filename in filenames[:nbQ]:
        loader, y_labels = create_data_loader(mode, f"{folder_name}/{filename}", all_possible_categories)
        loaders.append(loader)
        
    row_fixed, col_fixed = compute_row_col(nbQ, 3)
    
    df_datas = pd.read_excel(f"{folder_name}/{filenames[0]}", nrows=0)
    _, y_selected = get_y_and_labels(mode_selected, df_datas, False)
    
    return filenames, loaders, row_fixed, col_fixed, y_labels, y_selected
#
# beaucoup de repetition de code ici ...

def plot_predictions_and_targets_from_filenames(mode, mode_selected, model, nbQ, file_path, folder_name, num):
    # model learning not model_biorbd
    # pour plot une truc une colonne comme lmt, torque, etc

    filenames, loaders, row_fixed, col_fixed, y_labels, y_selected = general_plot_predictions(mode, mode_selected, folder_name, nbQ)

    fig, axs = plt.subplots(row_fixed,col_fixed, figsize=(15, 10))
    
    for q_index in range(nbQ) : 
        
        row = q_index // 3
        col = q_index % 3
        
        predictions, targets = get_predictions_and_targets_from_selected_y_labels(model, loaders[q_index], y_labels, y_selected)
        acc = mean_distance(torch.tensor(np.array(predictions)), torch.tensor(np.array(targets)))
        error_pourcentage = compute_pourcentage_error(torch.tensor(np.array(predictions)), torch.tensor(np.array(targets)))

        axs[row, col].plot(targets[:num], label='True values', marker='o', markersize=2)
        axs[row, col].plot(predictions[:num], label='Predictions', marker='D', linestyle='--', markersize=2)
        axs[row, col].set_title(f"File: {filenames[q_index].replace(".xlsx", "")}, acc = {acc:.6f}, error% = {error_pourcentage:.3f}%",fontsize='smaller')
        axs[row, col].set_xlabel(f'q{q_index} Variation',fontsize='smaller')
        axs[row, col].set_ylabel(f'{y_selected[0]}',fontsize='smaller')
        axs[row, col].legend()
    
    fig.suptitle(f'Predictions and targets of {y_selected[0]}', fontweight='bold')
    plt.tight_layout()  
    create_and_save_plot(f"{file_path}", f"plot_{y_selected[0]}_predictions_and_targets.png")
    plt.show()
    
    return None
        
def plot_predictions_and_targets_from_filenames_dlmt_dq(mode, mode_selected, model, nbQ, file_path, folder_name, num):

    filenames, loaders, row_fixed, col_fixed, y_labels, y_selected = general_plot_predictions(mode, mode_selected, folder_name, nbQ)
    
    # pour chaque q-index = 1 fig a chaque fois
    for q_index in range(nbQ) : 
        # on fait une nouvelle figure
        fig, axs = plt.subplots(row_fixed, col_fixed, figsize=(15, 10))
        # on recupere les predictions et targets de UN sheet --> 1 fig, len(q_ranges) plot
        predictions, targets = get_predictions_and_targets_from_selected_y_labels(model, loaders[q_index], y_labels, y_selected)
        
        if row_fixed == 1 and col_fixed == 1 : 
            acc = mean_distance(torch.tensor([prediction[0] for prediction in predictions]), torch.tensor([target[0] for target in targets]))
            error_pourcentage = compute_pourcentage_error(torch.tensor([prediction[0] for prediction in predictions]), torch.tensor([target[0] for target in targets]))
            
            plt.figure(figsize=(10, 5))
            plt.plot(targets[:num], label='True values', marker='o')
            plt.plot(predictions[:num], label='Predictions', marker='o',linestyle='--')
            plt.xlabel('q variation')
            plt.ylabel(f"{y_selected[0]}")
            plt.title(f"Predictions and targets of Lever Arm, q{q_index} variation, acc = {acc:.6f}, error% = {error_pourcentage:.3f}%", fontweight='bold')
            plt.legend()
        
        else : 
            # puis on fait chaque plot de la figure
            for i in range (len(y_selected)) : 
                acc = mean_distance(torch.tensor([prediction[i] for prediction in predictions]), torch.tensor([target[i] for target in targets]))
                error_pourcentage = compute_pourcentage_error(torch.tensor([prediction[i] for prediction in predictions]), torch.tensor([target[i] for target in targets]))
                
                # marche mais c,est moche :/
                row = i // 3
                col = i % 3
            
                axs[row, col].plot([target[i] for target in targets][:num], label='True values', marker='o', markersize=2)
                axs[row, col].plot([prediction[i] for prediction in predictions][:num], label='Predictions', marker='D', linestyle='--', markersize=2)
                axs[row, col].set_title(f"File: {filenames[q_index].replace(".xlsx", "")}, acc = {acc:.6f}, error% = {error_pourcentage:.3f}%",fontsize='smaller')
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
    
    plot_predictions_and_targets(model, y_labels, train_loader, "Train loader", 100, file_path, "train_loader")
    plot_predictions_and_targets(model, y_labels, val_loader, "Validation loader", 100, file_path, "val_loader")
    plot_predictions_and_targets(model, y_labels , test_loader, "Test loader", 100, file_path, "test_loader")

def visualize_prediction(mode, nbQ, file_path, folder_name_for_prediction) : 
    
    """
    Load saved model and plot-save visualisations 
    
    INPUTS 
    - mode : Mode
    - q_ranges : array, range of each qi (min,max)
    - y_labels : list string, labels of each type value in exit y
    - train_loader : DataLoader, data trainning (80% of 80%)
    - val_loader : DataLoader, data validation (20% of 80%)
    - test_loader : DataLoader, data testing (20%)
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

# -------------------------------------
def plot_mvt_discontinuities_in_red(i, qs, segment_lengths, to_remove) : 
    
    plt.plot(qs, segment_lengths, linestyle='-', color='b')
    qs_plot = [qs[idx] for idx in range(len(qs)) if idx not in to_remove]
    segment_lengths_plot = [segment_lengths[idx] for idx in range(len(segment_lengths)) if idx not in to_remove]
    plt.plot(qs_plot, segment_lengths_plot, marker='o', color='b')
    for idx in to_remove:
        plt.plot(qs[idx:idx+1], segment_lengths[idx:idx+1], marker='x', color='r')  # Discontinuities are in red
    plt.xlabel(f'q{i}')
    plt.ylabel('Muscle_length')
    plt.title(f'Muscle Length as a Function of q{i} Values')
    plt.xticks(qs[::5])
    plt.yticks(segment_lengths[::5]) 
    plt.grid(True)
    plt.show()


    
