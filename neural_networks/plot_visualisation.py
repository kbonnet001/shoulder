from math import ceil
import matplotlib as plt
import matplotlib.pyplot as plt
import torch
import numpy as np
import os
from neural_networks.data_preparation import create_data_loader
from neural_networks.file_directory_operations import create_and_save_plot
from neural_networks.other import compute_row_col

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

def plot_loss_and_accuracy(train_losses, val_losses, train_accs, val_accs, file_path):
    """Plot loss and accuracy (train and validation)

    INPUT :
    - train_losses :
    - val_losses
    - train_accs
    - val_accs """
    
    # Create subplots
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

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

# def plot_predictions_and_targets_ancienne(model, loader, string_loader, num, directory_path, loader_name) :
    
#     """Plot the true values and predicted values for a given model and data loader.
#     INPUT:
#     - model: The trained PyTorch model to be evaluated.
#     - loader: DataLoader containing the dataset to evaluate.
#     - num: The number of samples to plot for comparison.

#     OUTPUT:
#     - None: The function generates a plot showing the true values and predicted values.
#     """
    
#     predictions, targets = get_predictions_and_targets(model, loader)

#     acc = mean_distance(torch.tensor(predictions), torch.tensor(targets))
#     print("acc = ", acc)

#     # Plot
#     plt.figure(figsize=(10, 5))
#     plt.plot(targets[:num], label='True values', marker='o')
#     plt.plot(predictions[:num], label='Predictions', marker='o',linestyle='--')
#     plt.xlabel('Sample')
#     plt.ylabel('Muscle length')
#     plt.title(f"Predictions and targets - {string_loader}, acc = {acc:.6f}")
#     plt.legend()
#     create_and_save_plot(directory_path, f"plot_predictions_and_targets_{loader_name}")
#     plt.show()
    
def plot_predictions_and_targets(model, y_labels, loader, string_loader, num, directory_path, loader_name) :
    
    """Plot the true values and predicted values for a given model and data loader.
    INPUT:
    - model: The trained PyTorch model to be evaluated.
    - loader: DataLoader containing the dataset to evaluate.
    - num: The number of samples to plot for comparison.

    OUTPUT:
    - None: The function generates a plot showing the true values and predicted values.
    """
    num_rows = min(len(y_labels), 3)
    num_cols = max(1, (len(y_labels) + 1) // 3)
    
    predictions, targets = get_predictions_and_targets(model, loader)
    acc = mean_distance(torch.tensor(np.array(predictions)), torch.tensor(np.array(targets)))

    if num_cols == 1 and num_rows == 1 : 
        plt.figure(figsize=(10, 5))
        plt.plot(targets[:num], label='True values', marker='o')
        plt.plot(predictions[:num], label='Predictions', marker='o',linestyle='--')
        plt.xlabel('Sample')
        plt.ylabel(f"{y_labels[0]}")
        plt.title(f"Predictions and targets - {string_loader}, acc = {acc:.6f}", fontweight='bold')
        plt.legend()
    
    else :  
        accs = [mean_distance(torch.tensor(np.array(predictions[i])), torch.tensor(np.array(targets))) for i in range(len(y_labels))]
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 10))

        for k in range(len(y_labels)) : 
            
            row = k // ((len(y_labels) + 1) // 3)
            col = k % ((len(y_labels) + 1) // 3)
            
            axs[row, col].plot([arr[k] for arr in targets][:num], label='True values', marker='^')
            axs[row, col].plot([arr[k] for arr in predictions][:num], label='Predictions', marker='o',linestyle='--')
            axs[row, col].set_xlabel('Sample')
            axs[row, col].set_ylabel("Value")
            axs[row, col].set_title(f'{y_labels[k]}, acc = {accs[k]:.6f}',fontsize='smaller')
            axs[row, col].legend()
        
        fig.suptitle(f"Predictions and targets - {string_loader}", fontweight='bold')
        plt.tight_layout()  
    
    
    plt.savefig(f"{directory_path}/plot_predictions_and_targets.png")
    plt.show()


def plot_predictions_and_targets_from_filenames_muscle(mode, model, q_ranges, file_path, folder_name, num):

    all_possible_categories = [0,1,2,3,4,5,6,7,8,9,10,11]
    # on recupere les sheets et on les tris dans l'ordre
    filenames = sorted([filename for filename in os.listdir(folder_name)])
    # on fait des loaders pour chaque sheet
    loaders = [create_data_loader(mode, f"{folder_name}/{filename}", 0, all_possible_categories ) for filename in (filenames[:len(q_ranges)])]
    
    row_fixed, col_fixed = compute_row_col(len(q_ranges), 0, 3)
    fig, axs = plt.subplots(row_fixed,col_fixed, figsize=(15, 10))
    
    for q_index in range(len(q_ranges)) : 
        
        row = q_index // ((len(q_ranges) + 1) // 3)
        col = q_index % ((len(q_ranges) + 1) // 3)
        
        predictions, targets = get_predictions_and_targets(model, loaders[q_index])
        acc = mean_distance(torch.tensor(predictions), torch.tensor(targets))

        axs[row, col].plot(targets[:num], label='True values', marker='o')
        axs[row, col].plot(predictions[:num], label='Predictions', marker='D', linestyle='--')
        axs[row, col].set_title(f"File: {filenames[q_index].replace(".xlsx", "")}, acc = {acc:.6f}",fontsize='smaller')
        axs[row, col].set_xlabel(f'q{q_index} Variation',fontsize='smaller')
        axs[row, col].set_ylabel('Muscle_length (m)',fontsize='smaller')
        axs[row, col].legend()
    
    fig.suptitle(f'Predictions and targets of Muscle length', fontweight='bold')
    plt.tight_layout()  
    plt.savefig(f"{file_path}/plot_muscle_length_predictions_and_targets.png")
    plt.show()
    
    return None

    
    # for idx, loader in enumerate(loaders):
    #     row = idx // 2
    #     col = idx % 2
    #     ax = axs[row, col] if rows > 1 else axs[col]
        
    #     predictions, targets = get_predictions_and_targets(model, loader)
    #     acc = mean_distance(torch.tensor(predictions), torch.tensor(targets))
        
    #     ax.plot(targets[:num], label='True values', marker='o')
    #     ax.plot(predictions[:num], label='Predictions', marker='o', linestyle='--')
    #     ax.set_title(f"File: {filenames[idx]}, acc = {acc:.6f}")
    #     ax.set_xlabel('Sample')
    #     ax.set_ylabel('Muscle length')
    #     ax.legend()
        
        
def plot_predictions_and_targets_from_filenames_dlmt_dq(mode, model, y_labels, q_ranges, file_path, folder_name, num):

    all_possible_categories = [0,1,2,3,4,5,6,7,8,9,10,11]
    # on recupere les sheets et on les tris dans l'ordre
    filenames = sorted([filename for filename in os.listdir(folder_name)])
    # on fait des loaders pour chaque sheet
    loaders = [create_data_loader(mode, f"{folder_name}/{filename}", 0, all_possible_categories ) for filename in (filenames[:len(q_ranges)])]
    
    row_fixed, col_fixed = compute_row_col(len(q_ranges), 0, 3)
    
    # pour chaque q-index = 1 fig a chaque fois
    for q_index in range(len(q_ranges)) : 
        # on fait une nouvelle figure
        fig, axs = plt.subplots(row_fixed, col_fixed, figsize=(15, 10))
        # on recupere les predictions et targets de UN sheet --> 1 fig, len(q_ranges) plot
        predictions, targets = get_predictions_and_targets(model, loaders[q_index])
        # acc = mean_distance(torch.tensor(predictions), torch.tensor(targets))
        
        if row_fixed == 1 and col_fixed == 1 : 
            acc = mean_distance(torch.tensor([prediction[0] for prediction in predictions]), torch.tensor([target[0] for target in targets]))
                
            plt.figure(figsize=(10, 5))
            plt.plot(targets[:num], label='True values', marker='o')
            plt.plot(predictions[:num], label='Predictions', marker='o',linestyle='--')
            plt.xlabel('q variation')
            plt.ylabel(f"{y_labels[0]}")
            plt.title(f"Predictions and targets of Lever Arm, q{q_index} variation, acc = {acc:.6f}", fontweight='bold')
            plt.legend()
        
        else : 
            # puis on fait chaque plot de la figure
            for i in range (len(q_ranges)) : 
                acc = mean_distance(torch.tensor([prediction[i] for prediction in predictions]), torch.tensor([target[i] for target in targets]))
                
                # marche mais c,est moche :/
                row = i // ((len(q_ranges) + 1) // 3)
                col = i % ((len(q_ranges) + 1) // 3)
            
                axs[row, col].plot([target[i] for target in targets][:num], label='True values', marker='o', markersize=2)
                axs[row, col].plot([prediction[i] for prediction in predictions][:num], label='Predictions', marker='D', linestyle='--', markersize=2)
                axs[row, col].set_title(f"File: {filenames[q_index].replace(".xlsx", "")}, acc = {acc:.6f}",fontsize='smaller')
                axs[row, col].set_xlabel(f'q{q_index} Variation',fontsize='smaller')
                axs[row, col].set_ylabel(f'dlmt_dq{i}',fontsize='smaller')
                axs[row, col].legend()
        
            fig.suptitle(f'Predictions and targets of Lever Arm, q{q_index} variation', fontweight='bold')
            plt.tight_layout()  
            plt.savefig(f"{file_path}/q{q_index}_plot_lever_arm_predictions_and_targets.png")
            plt.show()
    
    return None

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
    

