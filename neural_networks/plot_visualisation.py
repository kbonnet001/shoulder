from math import ceil
import matplotlib as plt
import matplotlib.pyplot as plt
import torch
import numpy as np
import os
from neural_networks.data_preparation import create_data_loader, get_y_and_labels
from neural_networks.file_directory_operations import create_and_save_plot
from neural_networks.muscle_plotting_utils import compute_row_col, get_markers
from neural_networks.Mode import Mode
from neural_networks.save_model import load_saved_model
import pandas as pd
import matplotlib.colors as mcolors
import re

def mean_distance(predictions, targets):
    """
    Compute mean distance beetween predictions and targets

    Args :
    - predictions (torch.Tensor): Model's predictions 
    - targets (torch.Tensor): Targets

    Returns : 
        float: mean distance
    """
    distance = np.mean(np.abs(predictions - targets))
    return distance.item()

def compute_pourcentage_error(predictions, targets) : 
    """
    Compute mean relative error beetween predictions and targets

    Args :
    - predictions (torch.Tensor): Model's predictions 
    - targets (torch.Tensor): Targets

    Returns : 
        error_pourcentage : float, relative error
        error_pourcentage_abs : float, relative error in abs
    """
    # Security to avoid div by 0
    epsilon = np.finfo(float).eps
    targets[targets == 0] = epsilon
    
    error_pourcentage = np.mean((np.abs(predictions - targets)) / targets) * 100
    return error_pourcentage.item(), np.abs(error_pourcentage).item()

def plot_loss_and_accuracy(train_losses, val_losses, train_accs, val_accs, train_errors, val_errors, train_abs_errors, 
                               val_abs_errors, file_path, show_plot = False):
    """Plot loss, accuracy (train and validation), error percentage, and absolute error percentage (train and validation)

    This function visualizes the performance metrics of a model over epochs, including training and validation loss, 
    accuracy, percentage error, and absolute percentage error. It creates three subplots to present these metrics 
    separately and saves the plot to a specified file path.

    Args:
    - train_losses (list of float): All values of training loss recorded during training.
    - val_losses (list of float): All values of validation loss recorded during training.
    - train_accs (list of float): All values of training accuracy (mean distance) recorded during training.
    - val_accs (list of float): All values of validation accuracy (mean distance) recorded during training.
    - train_errors (list of float): All values of training percentage error recorded during training.
    - val_errors (list of float): All values of validation percentage error recorded during training.
    - train_abs_errors (list of float): All values of training absolute percentage error recorded during training.
    - val_abs_errors (list of float): All values of validation absolute percentage error recorded during training.
    - file_path (str): The file path where the plot will be saved.
    - show_plot (bool, optional): Whether to display the plot. Default is False.

    """
    
    # Create subplots
    _, axs = plt.subplots(1, 3, figsize=(15, 5))

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
    
    # Plot error pourcentage graph
    axs[2].plot(train_errors, label='Train Pourcentage Error')
    axs[2].plot(val_errors, label='Validation Pourcentage Error')
    axs[2].plot(train_abs_errors, label='Train Abs Pourcentage Error')
    axs[2].plot(val_abs_errors, label='Validation Abs Pourcentage Error')
    axs[2].set_xlabel('Epoch')
    axs[2].set_ylabel('Pourcentage error')
    axs[2].set_title('Train and Validation Pourcentage Error over Epochs')
    axs[2].legend()

    plt.tight_layout()
    
    create_and_save_plot(file_path, "plot_loss_and_accuracy")
    if show_plot == False : 
        plt.close()

# -----------------------------
def get_predictions_and_targets(model, data_loader, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    
    """Get predictions and targets from a model and data loader.

    Args:
    - model:The trained PyTorch model to be evaluated.
    - data_loader: DataLoader containing the dataset to evaluate.
    - device: The device to run the model on (default is CUDA if available, otherwise CPU).

    Returns:
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
            outputs = outputs.squeeze(1) # Remove dimensions of size 1
            labels = labels.squeeze(1) # Remove dimensions of size 1
            # Convert tensors to numpy arrays and extend the lists
            predictions.extend(outputs.cpu().numpy())
            targets.extend(labels.cpu().numpy())
    return predictions, targets
    
def plot_predictions_and_targets(model, y_labels, loader, string_loader, num, directory_path) :
    
    """Plot the true values and predicted values for a given model and data loader.
    
    Args:
    - model: The trained PyTorch model to be evaluated.
    - y_labels : [string], all y (outputs of model) names columns 
    - loader: DataLoader containing the dataset to evaluate.
    - string_loader : string, loader name
    - num: The number of samples to plot for comparison.
    - directory_path : string, path to save plot

    Returns:
    - None: The function generates a plot showing the true values and predicted values.
    """
    num_rows, num_cols = compute_row_col(len(y_labels))
    predictions, targets = get_predictions_and_targets(model, loader)
    
    # special case if nb_q == 1
    if num_cols == 1 and num_rows == 1 : 
        acc = mean_distance(np.array([predictions]), np.array([targets]))
        error_pourcen, error_pourcen_abs = compute_pourcentage_error(np.array([predictions]), np.array([targets]))
        
        plt.figure(figsize=(10, 5))
        plt.plot(targets[:num], label='True values', marker='o')
        plt.plot(predictions[:num], label='Predictions', marker='o',linestyle='--')
        plt.xlabel('Sample')
        plt.ylabel(f"{y_labels[0]}")
        plt.title(f"Predictions and targets - {string_loader}, acc = {acc:.6f}, error% = {error_pourcen:.3f}%, error abs% = {error_pourcen_abs:.3f}%", 
                  fontweight='bold')
        plt.legend()
    
    else :  
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 10))
        axs = axs.flatten() if num_rows == 1 or num_cols == 1 else axs

        for k in range(len(y_labels)) :
            row = k // num_cols
            col = k % num_cols
            index = k if num_rows == 1 or num_cols == 1 else (row, col)
            
            acc = mean_distance(np.array([prediction[k] for prediction in predictions]), 
                                np.array([target[k] for target in targets]))
            error_pourcen, error_pourcen_abs \
            = compute_pourcentage_error(np.array([prediction[k] for prediction in predictions]), 
                                        np.array([target[k] for target in targets]))
        
            axs[index].plot([target[k] for target in targets][:num], label='True values', marker='^', markersize=2)
            axs[index].plot([prediction[k] for prediction in predictions][:num], label='Predictions', marker='o',
                            linestyle='--', markersize=2)
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
    """Get predictions and targets for selected output columns from a model.

    This function evaluates a trained PyTorch model on a given dataset and extracts predictions and targets
    for specific output columns based on the provided y_labels and y_selected. It ensures that only the
    specified columns are returned, allowing for focused evaluation of particular model outputs.

    Args:
    - model (torch.nn.Module): The trained PyTorch model to be evaluated.
    - loader (DataLoader): DataLoader containing the dataset to evaluate.
    - y_labels (list of str): List of all possible output column names.
    - y_selected (list of str): List of selected output column names for evaluation.

    Returns:
    - selected_predictions (list of list): Predictions corresponding to the selected output columns.
    - selected_targets (list of list): Targets corresponding to the selected output columns.

    Raises:
    - ValueError: If any of the selected y_labels are not in the provided y_labels.
    """
    
    # Obtain predictions and targets using a helper function
    predictions, targets = get_predictions_and_targets(model, loader)
    
    # Check if all selected labels are valid
    for y in y_selected : 
        if y not in y_labels : 
            TypeError("error : y isn't in y_labels") 
            
    # If all labels are selected, return full predictions and targets
    if y_labels == y_selected : 
        return predictions, targets
            
    else : 
        # Find indices of the selected labels in y_labels
        selected_indices = [y_labels.index(label) for label in y_selected]
        
        # Select only the columns corresponding to the selected indices for both predictions and targets
        selected_predictions = [[row[i] for i in selected_indices] for row in predictions]
        selected_targets = [[row[i] for i in selected_indices] for row in targets]
        
        return selected_predictions, selected_targets

def general_plot_predictions(mode, batch_size, mode_selected, folder_name, total_subplot, nb_segment) :
    """Prepare data for plotting predictions by creating data loaders and determining subplot configuration.

    This function sets up data loaders for evaluating a model's predictions across multiple datasets stored in CSV files.
    It also computes the layout for subplots based on the number of datasets and returns labels for the outputs to
    be plotted.

    Args:
    - mode (Mode): The mode of the model, indicating the type of data being processed.
    - batch_size (int): The batch size for data loading.
    - mode_selected (Mode): The selected mode for plotting, which must be equal to or less comprehensive than `mode`.
        Examples: mode_selected = mode or mode = Mode.LMT_DLMT_DQ and mode_selected = MUSCLE
    - folder_name (str): Path to the folder containing CSV files with q variation data.
    - total_subplot (int): Number of total_subplot, indicating how many datasets to process.
    - nb_segment (int): Number of segments in the  biorbd model.

    Returns:
    - filenames (list of str): Names of each CSV file for q variations.
    - loaders (list of DataLoader): Data loaders for each dataset to be evaluated.
    - row_fixed (int): Number of rows for subplot layout.
    - col_fixed (int): Number of columns for subplot layout.
    - y_labels (list of str): Names of all output columns (y) from the model.
    - y_selected (list of str): Names of the selected output columns (y) to be plotted.
    """
    
    # Define all possible categories for one-hot encoding or label usage
    all_possible_categories = list(range(nb_segment))
    
    # Get all CSV filenames sorted from the specified folder
    filenames = sorted([filename for filename in os.listdir(folder_name)])
    filenames = [file for file in filenames if re.match(r'^\d', file)]
    
    # Create data loaders and retrieve output labels for each file up to nb_q
    loaders = []
    for filename in filenames:
        loader, y_labels = create_data_loader(mode, batch_size, f"{folder_name}/{filename}", all_possible_categories)
        loaders.append(loader)
        
    # Compute the number of rows and columns needed for subplots
    row_fixed, col_fixed = compute_row_col(total_subplot)
    
    # Load the first CSV to get the selected output labels for the mode_selected
    df_datas = pd.read_csv(f"{folder_name}/{filenames[0]}", nrows=0)
    _, y_selected = get_y_and_labels(mode_selected, df_datas, False)
    
    return filenames, loaders, row_fixed, col_fixed, y_labels, y_selected

def plot_predictions_and_targets_from_filenames(mode, mode_selected, model, batch_size, nb_q, nb_segment, nb_plot, file_path, 
                                                folder_name, num):
    """Create plots to compare predictions and targets for a specific column of y (e.g., lmt, torque).

    This function generates plots to visually compare the model's predictions with the true target values for each 
    dataset variation of q. The plots include accuracy and error metrics.

    Args:
    - mode (Mode): The mode of the model, indicating the type of data being processed.
    - mode_selected (Mode): The selected mode for plotting, which must be equal to or less comprehensive than `mode`.
        Examples: mode_selected = mode or mode = Mode.LMT_DLMT_DQ and mode_selected = MUSCLE
    - model (torch.nn.Module): The PyTorch model used for making predictions.
    - batch_size (int): The batch size for data loading.
    - nb_q (int): Number of q variations in the biorbd model, indicating how many datasets to process.
    - nb_segment (int): Number of segments in the  biorbd model.
    - file_path (str): Path to save the generated plot.
    - folder_name (str): Path to the folder containing CSV files with q variation data.
    - num (int): Number of data points to plot for each dataset.
    """
    # Prepare data and configurations for plotting using helper function
    filenames, loaders, row_fixed, col_fixed, y_labels, y_selected = general_plot_predictions(
        mode, batch_size, mode_selected, folder_name, nb_plot, nb_segment
    )
    
    # Iterate over each dataset to plot predictions and targets
    for q_index in range(nb_q):
        # Set up the figure and subplots for visualization
        fig, axs = plt.subplots(row_fixed, col_fixed, figsize=(15, 10))
        
        # Get selected predictions and targets from the data loader
        predictions, targets = get_predictions_and_targets_from_selected_y_labels(
            model, loaders[q_index], y_labels, y_selected
        )
        
        for i in range(nb_plot):
             
            acc = mean_distance(np.array([prediction[i] for prediction in predictions]), np.array([target[i] for target in targets]))
            error_pourcen, error_pourcen_abs = compute_pourcentage_error(np.array([prediction[i] for prediction in predictions]), 
                                                                            np.array([target[i] for target in targets]))
            
            row = i // col_fixed
            col = i % col_fixed
            
            axs[row, col].plot([target[i] for target in targets][:num], label='True values', marker='o', markersize=2)
            axs[row, col].plot([prediction[i] for prediction in predictions][:num], label='Predictions', marker='D', 
                                linestyle='--', markersize=2)
            axs[row, col].set_title(f"{y_selected[i]}\n"
                                    f"acc = {acc:.6f}, error% = {error_pourcen:.3f}%, error abs% = {error_pourcen_abs:.3f}%",
                                    fontsize='smaller')
            axs[row, col].set_xlabel(f'q{q_index} Variation', fontsize='smaller')
            axs[row, col].set_ylabel(f'{y_selected[i]}', fontsize='smaller')
            axs[row, col].legend()
            
        # Add a title to the entire figure and adjust the layout
        fig.suptitle(f'Predictions and targets of {str(mode).replace("Mode.", "")} File: {filenames[q_index].replace('.CSV', '')}', 
                    fontweight='bold')
        plt.tight_layout()  
        
        # Save the plot to the specified file path
        create_and_save_plot(file_path, f"plot_{q_index}_{str(mode).replace("Mode.", "")}_predictions_and_targets.png")
        plt.show()

# def plot_predictions_and_targets_from_filenames_dlmt_dq(mode, mode_selected, model, batch_size, nb_q, nb_segment, file_path, folder_name, num):
#     """Create plots to compare predictions and targets for DLMT_DQ.

#     This function generates plots for each dataset variation of q, comparing the model's predictions with the true target values.
#     Special handling is applied if there's only one dataset.

#     Args:
#     - mode (Mode): Mode of the model, indicating the type of data.
#     - mode_selected (Mode): Selected mode for plotting, which must be equal to or less comprehensive than `mode`.
#         Examples: mode_selected = mode or mode = Mode.LMT_DLMT_DQ and mode_selected = MUSCLE
#     - model (torch.nn.Module): PyTorch model used for predictions.
#     - batch_size (int): Batch size for data loading.
#     - nb_q (int): Number of q variations in the biorbd model.
#     - nb_segment (int): Number of segments in the  biorbd model.
#     - file_path (str): Path to save the plots.
#     - folder_name (str): Path to the folder containing CSV files with q variation data.
#     - num (int): Number of data points to plot.
#     """
    
#     filenames, loaders, row_fixed, col_fixed, y_labels, y_selected = general_plot_predictions(
#         mode, batch_size, mode_selected, folder_name, nb_q, nb_segment
#     )
    
#     # Generate plots for each q variation
#     for q_index in range(nb_q):
#         if row_fixed == 1 and col_fixed == 1:
#             # Special case for a single dataset
#             predictions, targets = get_predictions_and_targets_from_selected_y_labels(
#                 model, loaders[q_index], y_labels, y_selected
#             )
            
#             # Compute accuracy and error metrics
#             acc = mean_distance(np.array([prediction[0] for prediction in predictions]), np.array([target[0] for target in targets]))
#             error_pourcen, error_pourcen_abs = compute_pourcentage_error(np.array([prediction[0] for prediction in predictions]), 
#                                                                          np.array([target[0] for target in targets]))
            
#             plt.figure(figsize=(10, 5))
#             plt.plot(targets[:num], label='True values', marker='o')
#             plt.plot(predictions[:num], label='Predictions', marker='o', linestyle='--')
#             plt.xlabel('q variation')
#             plt.ylabel(f"{y_selected[0]}")
#             plt.title(f"Predictions and targets of Lever Arm, q{q_index} variation\n"
#                       f"acc = {acc:.6f}, error% = {error_pourcen:.3f}%, error abs% = {error_pourcen_abs:.3f}%", 
#                       fontweight='bold')
#             plt.legend()
#             create_and_save_plot(file_path, f"q{q_index}_plot_length_jacobian_predictions_and_targets.png")
#             plt.show()
        
#         else:
#             # For multiple datasets, create a subplot for each y_selected
#             fig, axs = plt.subplots(row_fixed, col_fixed, figsize=(15, 10))
            
#             predictions, targets = get_predictions_and_targets_from_selected_y_labels(
#                 model, loaders[q_index], y_labels, y_selected
#             )
            
#             for i in range(len(y_selected)):
#                 acc = mean_distance(np.array([prediction[i] for prediction in predictions]), np.array([target[i] for target in targets]))
#                 error_pourcen, error_pourcen_abs = compute_pourcentage_error(np.array([prediction[i] for prediction in predictions]), 
#                                                                              np.array([target[i] for target in targets]))
                
#                 row = i // col_fixed
#                 col = i % col_fixed
                
#                 axs[row, col].plot([target[i] for target in targets][:num], label='True values', marker='o', markersize=2)
#                 axs[row, col].plot([prediction[i] for prediction in predictions][:num], label='Predictions', marker='D', 
#                                    linestyle='--', markersize=2)
#                 axs[row, col].set_title(f"File: {filenames[q_index].replace('.CSV', '')}\n"
#                                         f"acc = {acc:.6f}, error% = {error_pourcen:.3f}%, error abs% = {error_pourcen_abs:.3f}%",
#                                         fontsize='smaller')
#                 axs[row, col].set_xlabel(f'q{q_index} Variation', fontsize='smaller')
#                 axs[row, col].set_ylabel(f'dlmt_dq{i}', fontsize='smaller')
#                 axs[row, col].legend()
            
#             fig.suptitle(f'Predictions and targets of Lever Arm, q{q_index} variation', fontweight='bold')
#             plt.tight_layout()
#             create_and_save_plot(file_path, f"q{q_index}_plot_length_jacobian_predictions_and_targets.png")
#             plt.show()
            

def plot_predictions_and_targets_from_filenames_dlmt_dq(mode, mode_selected, model, batch_size, nb_q, nb_segment, nb_muscle, file_path, folder_name, num):
    """Create plots to compare predictions and targets for DLMT_DQ.

    This function generates plots for each dataset variation of q, comparing the model's predictions with the true target values.
    Special handling is applied if there's only one dataset.

    Args:
    - mode (Mode): Mode of the model, indicating the type of data.
    - mode_selected (Mode): Selected mode for plotting, which must be equal to or less comprehensive than `mode`.
        Examples: mode_selected = mode or mode = Mode.LMT_DLMT_DQ and mode_selected = MUSCLE
    - model (torch.nn.Module): PyTorch model used for predictions.
    - batch_size (int): Batch size for data loading.
    - nb_q (int): Number of q variations in the biorbd model.
    - nb_muscle (int): Number of muscles in the biorbd model.
    - nb_segment (int): Number of segments in the  biorbd model.
    - file_path (str): Path to save the plots.
    - folder_name (str): Path to the folder containing CSV files with q variation data.
    - num (int): Number of data points to plot.
    
    Warning : nb_q != 1 
    """
    
    filenames, loaders, row_fixed, col_fixed, y_labels, y_selected = general_plot_predictions(
        mode, batch_size, mode_selected, folder_name, nb_muscle, nb_segment
    )
    
    # Generate plots for each q variation
    for q_index in range(nb_q): 
        # For multiple datasets, create a subplot for each y_selected
        fig, axs = plt.subplots(row_fixed, col_fixed, figsize=(20, 10))
        
        predictions, targets = get_predictions_and_targets_from_selected_y_labels(model, loaders[q_index], y_labels,
                                                                                    y_selected)
        colors = plt.cm.gist_rainbow(np.linspace(0, 1, nb_q))
        colors = [tuple(c * 0.8 for c in mcolors.to_rgba(color)[:3]) + (mcolors.to_rgba(color)[3],) for color in colors]
        markers = get_markers(nb_q)
        
        for muscle_index in range(nb_muscle) : 
            row = muscle_index // col_fixed
            col = muscle_index % col_fixed
                
            acc = mean_distance(np.array([prediction[nb_q*muscle_index : nb_q*muscle_index+8] for prediction in predictions]), np.array([target[nb_q*muscle_index : nb_q*muscle_index+8] for target in targets]))
            error_pourcen, error_pourcen_abs = compute_pourcentage_error(np.array([prediction[nb_q*muscle_index : nb_q*muscle_index+8] for prediction in predictions]), 
                                                                            np.array([target[nb_q*muscle_index : nb_q*muscle_index+8] for target in targets]))
            
            for i in range(nb_q):     
                color = colors[i]
                special_marker = markers[i]  # Market for the first point
                
                # data prediction and targets
                target_data = [target[nb_q * muscle_index + i] for target in targets][:num]
                prediction_data = [prediction[nb_q * muscle_index + i] for prediction in predictions][:num]
                
                # First point target
                axs[row, col].plot(range(num), target_data, 
                                color="gray", alpha=0.5, marker='o', markersize=2)

                # Target
                axs[row, col].scatter(0, target_data[0], label=f"Targ (q{i+1})" if muscle_index == 0 else None,
                                    color="gray", marker=special_marker, s=64, edgecolor='black', alpha=0.6)
                
                # First point prediction
                axs[row, col].plot(range(num), prediction_data, 
                                linestyle='--', color=color, marker='o', markersize=2)

                # Prediction
                axs[row, col].scatter(0, prediction_data[0], label=f'Pred (q{i+1})' if muscle_index == 0 else None, 
                                    color=color, marker=special_marker, edgecolor='black', s=64)
            
                # Legend only on the first subplot
                if muscle_index == 0:
                    axs[row, col].legend(loc='upper right', handletextpad=0.5, fontsize=6)

                    
            axs[row, col].set_title(f"Muscle: {muscle_index}, acc = {acc:.3e},\n error% = {error_pourcen:.3e}%, error abs% = {error_pourcen_abs:.3e}%",
                                    fontsize='smaller', fontweight='bold')
            axs[row, col].set_xlabel(f'q{q_index} Variation', fontsize='smaller')
            axs[row, col].set_ylabel(f'dlmt_dq{muscle_index}', fontsize='smaller')
    
        plt.tight_layout()
        plt.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.75, hspace=0.2, wspace=0.2)

        # Title wax removed because it was hided by subplot ...
        # fig.suptitle(f"Predictions and targets of Lever Arm, {filenames[q_index].replace('.CSV', '')}", fontweight='bold')
        plt.tight_layout()
        create_and_save_plot(file_path, f"{filenames[q_index].replace('.CSV', '')}_plot_length_jacobian_predictions_and_targets.png")
        plt.show()

# -------------------------------------
def visualize_prediction_training(model, file_path, y_labels, train_loader, val_loader, test_loader, limit=16):
    """Create plots of predictions and targets for training, validation, and test datasets.

    This function generates and saves plots comparing model predictions to true target values for training,
    validation, and test datasets.

    Args:
    - model (torch.nn.Module): The PyTorch model to evaluate.
    - file_path (str): Path to save the plots.
    - y_labels (list of str): Names of the columns (outputs of the model) to visualize.
    - train_loader (DataLoader): DataLoader for the training data.
    - val_loader (DataLoader): DataLoader for the validation data.
    - test_loader (DataLoader): DataLoader for the test data.
    - limit (int): defalt 16, limit of subplot to avoid bug
    """
    
    # Security to avoid bug, to many subplot to plot
    if len(y_labels) > limit : 
        y_labels = y_labels[-limit:] # last labels
    
    # Generate and save plots for each dataset type
    plot_predictions_and_targets(model, y_labels, train_loader, "Train loader", 100, file_path)
    plot_predictions_and_targets(model, y_labels, val_loader, "Validation loader", 100, file_path)
    plot_predictions_and_targets(model, y_labels, test_loader, "Test loader", 100, file_path)

def visualize_prediction(mode, batch_size, nb_q, nb_segment, nb_muscle, file_path, folder_name_for_prediction):
    """
    Load a saved model and generate visualizations for predictions and targets based on the mode.

    Args:
    - mode (Mode): The mode indicating which type of predictions and plots to generate.
    - batch_size (int): The batch size used for predictions.
    - nb_q (int): Number of q in the biorbd model.
    - nb_segment (int): Number of segments in the  biorbd model.
    - nb_muscle (int): Number of muscles in the  biorbd model.
    - file_path (str): Path to the directory containing the 'model_config.json' file.
    - folder_name_for_prediction (str): Path to the folder containing files for plotting all q variations.
    """
    
    # Load the saved model
    model = load_saved_model(file_path)
    
    # Generate plots based on the mode
    if mode == Mode.DLMT_DQ:
        plot_predictions_and_targets_from_filenames_dlmt_dq(
            mode, mode, model, batch_size, nb_q, nb_segment, nb_muscle, file_path, folder_name_for_prediction, 100
        )
    
    elif mode == Mode.MUSCLE_DLMT_DQ:
        plot_predictions_and_targets_from_filenames(
            mode, Mode.MUSCLE, model, batch_size, nb_q, nb_segment, nb_q, file_path, folder_name_for_prediction, 100
        )
        plot_predictions_and_targets_from_filenames_dlmt_dq(
            mode, Mode.DLMT_DQ, model, batch_size, nb_q, nb_segment, nb_muscle, file_path, folder_name_for_prediction, 100
        )
    
    elif mode == Mode.TORQUE_MUS_DLMT_DQ:
        plot_predictions_and_targets_from_filenames(
            mode, Mode.MUSCLE, model, batch_size, nb_q, nb_segment, nb_muscle, file_path, folder_name_for_prediction, 100
        )
        plot_predictions_and_targets_from_filenames_dlmt_dq(
            mode, Mode.DLMT_DQ, model, batch_size, nb_q, nb_segment, nb_muscle, file_path, folder_name_for_prediction, 100
        )
        plot_predictions_and_targets_from_filenames(
            mode, Mode.TORQUE, model, batch_size, nb_q, nb_segment, nb_q, file_path, folder_name_for_prediction, 100
        )
    
    elif mode == Mode.DLMT_DQ_FM:
        plot_predictions_and_targets_from_filenames_dlmt_dq(
            mode, Mode.DLMT_DQ, model, batch_size, nb_q, nb_segment, nb_muscle, file_path, folder_name_for_prediction, 100
        )
        plot_predictions_and_targets_from_filenames(
            mode, Mode.FORCE, model, batch_size, nb_q, nb_segment, nb_muscle, file_path, folder_name_for_prediction, 100
        )
    
    elif mode == Mode.MUSCLE:
        plot_predictions_and_targets_from_filenames(
            mode, mode, model, batch_size, nb_q, nb_segment, nb_muscle, file_path, folder_name_for_prediction, 100
        )
    elif mode ==  Mode.TORQUE:
        plot_predictions_and_targets_from_filenames(
            mode, mode, model, batch_size, nb_q, nb_segment, nb_q, file_path, folder_name_for_prediction, 100
        )
    elif mode == Mode.DLMT_DQ_F_TORQUE : 
        plot_predictions_and_targets_from_filenames_dlmt_dq(
            mode, Mode.DLMT_DQ, model, batch_size, nb_q, nb_segment, nb_muscle, file_path, folder_name_for_prediction, 100
        )
        plot_predictions_and_targets_from_filenames(
            mode, Mode.FORCE, model, batch_size, nb_q, nb_segment, nb_muscle, file_path, folder_name_for_prediction, 100
        )
        plot_predictions_and_targets_from_filenames(
            mode, Mode.TORQUE, model, batch_size, nb_q, nb_segment, nb_q, file_path, folder_name_for_prediction, 100
        )
    
    else:
        # Raise an error if the mode is not valid
        raise ValueError(f"Invalid mode: {mode}. The mode does not exist or is not supported.")
