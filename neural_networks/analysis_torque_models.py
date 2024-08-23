from neural_networks.data_preparation import get_x, get_y_and_labels
from neural_networks.save_model import main_function_model
import pandas as pd
import torch
from neural_networks.Mode import Mode
import copy
from neural_networks.muscle_forces_and_torque import compute_torque
import numpy as np
import matplotlib.pyplot as plt
from neural_networks.file_directory_operations import create_and_save_plot
from neural_networks.plot_visualisation import mean_distance, compute_pourcentage_error
import os
from neural_networks.muscle_plotting_utils import compute_row_col

def compare_model_torque_prediction(save_model_paths, modes, nb_q, csv_path_datas, save_file_path, num=100):
    """
    This function compares torque predictions from different models with the actual torque data.
    The comparison is based on data from CSV files, and the results are plotted and saved.
    Warning : This function probably needs some adjustment, especially to better present accuracy, percentage error, 
    and absolute percentage error.

    Args:
    - save_model_paths: List of file paths to the saved models to be used for predictions.
    - modes: List of Mode that define how to process the data (e.g., TORQUE, DLMT_DQ_F_TORQUE, DLMT_DQ_FM).
    - nb_q: Number of torque components (typically joint angles).
    - csv_path_datas: List of paths to the CSV files containing the data.
    - save_file_path: Path where the plot images will be saved.
    - num: Number of data points to plot.

    Returns:
    - None. The function plots the results and saves the images.
    """

    # Initialize lists to store predictions and labels
    torque_predictions = []
    labels = []

    # Calculate the number of rows and columns for subplots based on nb_q
    row_fixed, col_fixed = compute_row_col(nb_q)
    
    # Loop through each CSV file in the provided paths
    for file in csv_path_datas:
        # Create a figure and subplots
        fig, axs = plt.subplots(row_fixed, col_fixed, figsize=(15, 10))
        torque_predictions = []
        labels = []
        accs_errors = []
        
        # Load data from the CSV file
        df_datas = pd.read_csv(file)
        selected_columns_torque = [col for col in df_datas.columns if col.startswith('torque_')]
        torque_columns = df_datas.filter(like="torque_").columns
        
        # Extract the file name (without extension) from the path
        file_name = os.path.basename(file)
        file_name = os.path.splitext(file_name)[0]

        # Loop through each model and mode
        for save_model, mode in zip(save_model_paths, modes):
            # Retrieve input data based on the mode
            x = get_x(mode, df_datas, get_origin_and_insertion=False)
            
            # Convert input data to PyTorch tensors
            inputs = torch.tensor(x, dtype=torch.float32)
            
            # Get labels for the output data
            _, y_labels = get_y_and_labels(mode, df_datas, get_y=False)
            
            # Run the model and get the outputs
            outputs, _, _ = main_function_model(save_model, inputs)
            outputs = outputs.detach().numpy()
            
            # Store the model label for later use in plots
            labels.append(save_model.split("_Model/", 1)[1])
            
            # Extract the target torque values from the CSV data
            targets = df_datas.loc[:, selected_columns_torque].values
            
            if mode == Mode.TORQUE:
                # If the mode is TORQUE, the outputs already contain torque predictions
                acc = mean_distance(outputs, targets)
                error, abs_error = compute_pourcentage_error(outputs, targets)
                
                # Store predictions and errors
                torque_predictions.append(outputs)
                accs_errors.append([acc, error, abs_error])
                
            elif mode == Mode.DLMT_DQ_F_TORQUE:
                # If the mode is DLMT_DQ_F_TORQUE, extract torque predictions from outputs
                torque_columns = [i for i, label in enumerate(y_labels) if label.startswith("torque_")]
                torque = outputs[:, torque_columns]
                acc = mean_distance(torque, targets)
                error, abs_error = compute_pourcentage_error(torque, targets)
                
                # Store predictions and errors
                torque_predictions.append(torque)
                accs_errors.append([acc, error, abs_error])
                
            elif mode == Mode.DLMT_DQ_FM:
                # If the mode is DLMT_DQ_FM, torque needs to be calculated manually
                dlmt_dq_columns = [i for i, label in enumerate(y_labels) if label.startswith("dlmt_dq_")]
                fm_columns = [i for i, label in enumerate(y_labels) if label.startswith("muscle_force_")]
                
                # Extract the relevant columns from outputs
                dlmt_dq_flat = outputs[:, dlmt_dq_columns]
                
                # Reshape the matrix to have nb_q columns per row
                if dlmt_dq_flat.shape[1] % nb_q != 0:
                    raise ValueError("dlmt_dq_columns must be a multiple of nb_q")
                dlmt_dq = dlmt_dq_flat.reshape(-1, dlmt_dq_flat.shape[1] // nb_q, nb_q)
                fm = outputs[:, fm_columns]
                
                # Compute torque manually
                torque = [compute_torque(dlmt_dq_i, fm_i) for dlmt_dq_i, fm_i in zip(dlmt_dq, fm)]
                acc = mean_distance(torque, targets)
                error, abs_error = compute_pourcentage_error(torque, targets)
                
                # Store predictions and errors
                torque_predictions.append(np.array(torque))
                accs_errors.append([acc, error, abs_error])
                
            else:
                # Raise an error if the mode is not recognized
                raise ValueError(f"Invalid mode: {mode}. The mode does not exist or is not supported.")
    
        # Plot the results
        for i in range(nb_q):
            row = i // col_fixed
            col = i % col_fixed
            
            # Plot the target torque values
            axs[row, col].plot([target[i] for target in targets][:num], marker='o', markersize=2, 
                               label="torque_target")
            
            # Plot the predicted torque values from each model
            for k in range (len(torque_predictions)) : 
                predictions = torque_predictions[k]
                # acc = mean_distance(np.array([prediction[i] for prediction in predictions]), 
                #                     np.array([target[i] for target in targets]))
                # error, abs_error = compute_pourcentage_error(np.array([prediction[i] for prediction in predictions]), 
                #                                             np.array([target[i] for target in targets]))
                
                axs[row, col].plot([prediction[i] for prediction in predictions][:num], marker='o', linestyle='-',
                                   markersize=2, label = f"{labels[k]}" ) #acc = {acc:.3e},\nerror% = {error:.3e}, abs_error% = {abs_error:.3e}"
                
            # Set title and labels for each subplot
            axs[row, col].set_title(f"{file_name.replace('.CSV', '')}", fontsize='smaller') 
            axs[row, col].set_xlabel(f'q{i} Variation', fontsize='smaller')
            axs[row, col].set_ylabel(f'Torque Nm', fontsize='smaller')
            axs[row, col].legend()
        
        # Print the accuracy and error information for each model
        print(f"-----------------------\nlabel_model: [acc, error%, abs_error%]\n")
        for k in range(len(torque_predictions)):
            print(f"{labels[k]}: {accs_errors[k]}\n")
        
        # Add a title to the entire figure and adjust the layout
        fig.suptitle(f'Predictions and targets - Torque File: {file_name.replace(".CSV", "")}', fontweight='bold')
        plt.tight_layout()
        
        # Save the plot to the specified file path
        create_and_save_plot(save_file_path, f"plot_{file_name.replace('.CSV', '')}_predictions_and_targets_torque.png")
        plt.show()
