from math import ceil
import matplotlib as plt
import matplotlib.pyplot as plt
import torch
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

def plot_predictions_and_targets_from_filenames(model, filenames, limits, num):

    loaders = [create_data_loader(filename, limit) for filename, limit in zip(filenames, limits)]

    num_files = len(loaders)
    rows = ceil(num_files / 2)
    
    fig, axs = plt.subplots(rows, 2, figsize=(20, 5 * rows))
    
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
    

