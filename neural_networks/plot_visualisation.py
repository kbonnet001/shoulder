from math import ceil
import matplotlib as plt
import matplotlib.pyplot as plt
import torch
import os
from neural_networks.data_preparation import create_data_loader
from neural_networks.file_directory_operations import create_and_save_plot

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

    
def plot_predictions_and_targets(model, loader, string_loader, num, directory_path, loader_name) :
    
    """Plot the true values and predicted values for a given model and data loader.
    INPUT:
    - model: The trained PyTorch model to be evaluated.
    - loader: DataLoader containing the dataset to evaluate.
    - num: The number of samples to plot for comparison.

    OUTPUT:
    - None: The function generates a plot showing the true values and predicted values.
    """
    
    predictions, targets = get_predictions_and_targets(model, loader)

    acc = mean_distance(torch.tensor(predictions), torch.tensor(targets))
    print("acc = ", acc)

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(targets[:num], label='True values', marker='o')
    plt.plot(predictions[:num], label='Predictions', marker='o',linestyle='--')
    plt.xlabel('Échantillons')
    plt.ylabel('Muscle length')
    plt.title(f"Predictions and targets - {string_loader}, acc = {acc:.6f}")
    plt.legend()
    create_and_save_plot(directory_path, f"plot_predictions_and_targets_{loader_name}")
    plt.show()

def plot_predictions_and_targets_from_filenames(model, q_ranges, file_path, folder_name, num):

    all_possible_categories = [0,1,2,3,4,5,6,7,8,9,10,11]
    filenames = sorted([filename for filename in os.listdir(folder_name)])
    loaders = [create_data_loader(f"{folder_name}/{filename}", 0, all_possible_categories ) for filename in (filenames[:len(q_ranges)])]
    
    fig, axs = plt.subplots(3, (len(q_ranges)+1)//3, figsize=(15, 10))
    
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
    
    fig.suptitle(f'Predictions and targets of Muscle length Muscle', fontweight='bold')
    plt.tight_layout()  
    plt.savefig(f"{file_path}/plot_muscle_length_predictions_and_targets.png")
    plt.show()
    
    return None

    

    
    for idx, loader in enumerate(loaders):
        row = idx // 2
        col = idx % 2
        ax = axs[row, col] if rows > 1 else axs[col]
        
        predictions, targets = get_predictions_and_targets(model, loader)
        acc = mean_distance(torch.tensor(predictions), torch.tensor(targets))
        
        ax.plot(targets[:num], label='True values', marker='o')
        ax.plot(predictions[:num], label='Predictions', marker='o', linestyle='--')
        ax.set_title(f"File: {filenames[idx]}, acc = {acc:.6f}")
        ax.set_xlabel('Échantillons')
        ax.set_ylabel('Muscle length')
        ax.legend()

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
    

