import matplotlib as plt
import matplotlib.pyplot as plt
import numpy as np
import os
from neural_networks.file_directory_operations import create_and_save_plot, read_info_model
import pandas as pd
from neural_networks.CSVBatchWriterTestHyperparams import CSVBatchWriterTestHyperparams
from neural_networks.muscle_plotting_utils import get_markers

def find_points_front_pareto(num_points, x_axis, y_axis):
    """
    Identifies indices of points on the Pareto front from given x and y coordinates.

    Args:
        num_points (int): The number of points to evaluate.
        x_axis (list): List of x-coordinates for the points.
        y_axis (list): List of y-coordinates for the points.

    Returns:
        list: Indices of points that are part of the Pareto front.
    """
    pareto_indices = []  # List to store indices of non-dominated points

    # Iterate over each point to check if it is dominated by any other point
    for i in range(num_points):
        dominated = False  # Flag to check if the current point is dominated

        # Compare the current point with all other points
        for j in range(num_points):
            # Check if point j dominates point i
            if (x_axis[j] <= x_axis[i] and y_axis[j] <= y_axis[i]) and (x_axis[j] < x_axis[i] or y_axis[j] < y_axis[i]):
                dominated = True  # Point i is dominated, no need to check further
                break

        # If not dominated, add the index of the current point to the Pareto front
        if not dominated:
            pareto_indices.append(i)

    return pareto_indices  # Return the indices of the Pareto front points

def plot_results_try_hyperparams(csv_file, x_info, y_info, id="num_try"):
    """
    Plots the results from a CSV file and highlights the Pareto front.

    Args:
        csv_file (str): Path to the CSV file containing the data.
        x_info (str): Column name for the x-axis data.
        y_info (str): Column name for the y-axis data.
        id (str, optional): Column name for the model identifier. Defaults to 'num_try'.
    """

    # Check if the CSV file exists and load the data
    if os.path.exists(csv_file):
        df_datas = pd.read_csv(csv_file)
        x_axis = df_datas.loc[:, x_info].values  # Extract x-axis data
        y_axis = df_datas.loc[:, y_info].values  # Extract y-axis data
        model_id = df_datas.loc[:, id].values    # Extract model identifiers

    # Generate unique colors for each point using a colormap
    num_points = len(x_axis)
    colors = plt.cm.jet(np.linspace(0, 1, num_points))  # Use the jet colormap for diversity

    # Configure plot size and scale
    plt.figure(figsize=(10, 5))
    plt.xscale('log')
    plt.yscale('log')

    # Detect Pareto front points
    pareto_indices = find_points_front_pareto(num_points, x_axis, y_axis)

    # Plot each point and highlight Pareto front points
    for i in range(num_points):
        plt.scatter(x_axis[i], y_axis[i], marker='P', color=colors[i])  # Plot each point

        # Highlight and annotate Pareto front points
        if i in pareto_indices:
            plt.scatter(x_axis[i], y_axis[i], edgecolor='black', facecolor='none', s=100)
            plt.text(x_axis[i], y_axis[i], model_id[i], fontsize=9, ha='right', weight='bold')
        else:
            plt.text(x_axis[i], y_axis[i], model_id[i], fontsize=9, ha='right')

    # Draw a line connecting Pareto front points
    pareto_points = sorted([(x_axis[i], y_axis[i]) for i in pareto_indices])
    pareto_x, pareto_y = zip(*pareto_points)  # Unzip into x and y components
    plt.plot(pareto_x, pareto_y, linestyle='--', color='black', alpha=0.6, label="Pareto_front")

    # Annotate the best solutions along the Pareto front
    plt.text(pareto_x[0], pareto_y[0] * 1.05, f"Best solution\n of objective\n '{x_info}'", fontsize=9, ha='left', 
             va='top', bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))
    plt.text(pareto_x[-1] * 1.05, pareto_y[-1], f"Best solution\n of objective\n '{y_info}'", fontsize=9, ha='right', 
             va='bottom', bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))

    # Label axes and set plot title
    plt.xlabel(f"{x_info}" + (" (s)" if x_info.startswith("execution_time") else ""))
    plt.ylabel(f"{y_info}" + (" (s)" if y_info.startswith("execution_time") else ""))
    plt.title(f"{x_info} vs {y_info}", fontweight='bold')

    plt.grid(True)
    plt.legend()

    # Save the plot to the specified directory
    create_and_save_plot(f"{os.path.dirname(csv_file)}", f"{x_info} vs {y_info}.png")
    plt.show()  # Display the plot

def plot_results_try_hyperparams_comparaison(dir_paths, x_info, y_info, save_path, id="num_try"):
    """
    Plots and compares results from multiple directories containing CSV files, highlighting differences.

    Args:
        dir_paths (list): List of directory paths to CSV files for comparison.
        x_info (str): Column name for the x-axis data.
        y_info (str): Column name for the y-axis data.
        save_path (str): Path where the plot will be saved.
        id (str, optional): Column name for the model identifier. Defaults to 'num_try'.
    """

    x_axis = []  # List to store x-axis data from each file
    y_axis = []  # List to store y-axis data from each file
    model_id = []  # List to store model identifiers from each file

    # Iterate through each directory path and load the data
    for full_directory_path in dir_paths:
        if os.path.exists(f"{full_directory_path}"):
            df_datas = pd.read_csv(full_directory_path)
            x_axis.append(df_datas.loc[:, y_info].values)  # Append y_info data as x-axis
            y_axis.append(df_datas.loc[:, x_info].values)  # Append x_info data as y-axis
            model_id.append(df_datas.loc[:, id].values)    # Append model identifiers

    # Generate unique colors and markers for each set of data
    num_colors = len(dir_paths)
    colors = plt.cm.jet(np.linspace(0, 1, num_colors))  # Use jet colormap
    markers = get_markers(len(dir_paths))  # Different markers for each dataset

    # Configure plot size and scale
    plt.figure(figsize=(10, 5))
    plt.xscale('log')
    plt.yscale('log')

    # Extract file names for legend labels
    file_names = [path.split('/')[-1] for path in dir_paths]

    # Plot each set of points with its respective color and marker
    for i in range(len(dir_paths)):
        for j in range(len(x_axis[i])):
            if j == 0:  # Add label only to the first point of each dataset for legend
                plt.scatter(y_axis[i][j], x_axis[i][j], marker=markers[i], color=colors[i], label=file_names[i])
            else:
                plt.scatter(y_axis[i][j], x_axis[i][j], marker=markers[i], color=colors[i])

            # Annotate each point with its model ID
            plt.text(y_axis[i][j], x_axis[i][j], model_id[i][j], fontsize=9, ha='right')

    # Set axis labels and plot title
    plt.xlabel(f"{x_info}" + (" (s)" if x_info == "execution_time" else ""))
    plt.ylabel(f"{y_info}" + (" (s)" if y_info == "execution_time" else ""))
    plt.title(f"{x_info} vs {y_info}", fontweight='bold')

    plt.grid(True)
    plt.legend()

    # Construct filename for saving the plot
    text_file_names = " vs ".join([file_name.split('.')[0] for file_name in file_names])
    create_and_save_plot(f"{save_path}", f"{text_file_names}.png")

    plt.show()  # Display the plot

def create_df_from_txt_saved_informations(directory_csv):
    """
    Creates a DataFrame from information stored in text files within directories
    and writes the data to a CSV file using batch processing.

    Args:
        directory_csv (str): Path to the CSV file where data will be written.
    """

    # Initialize a CSV writer with a specified batch size
    writer = CSVBatchWriterTestHyperparams(f"{directory_csv}", batch_size=100)

    # Iterate over directories in the parent directory of the CSV path
    for directory in os.listdir(f"{os.path.dirname(directory_csv)}"):
        full_directory_path = os.path.join(f"{os.path.dirname(directory_csv)}", directory)

        # Check if the directory contains a model information text file
        if os.path.isdir(full_directory_path) and os.path.exists(f"{full_directory_path}/model_informations.txt"):
            # Read model information from the text file
            num_try, val_loss, val_acc, train_timer, mean_model_load_timer, \
            mean_model_timer, mode, batch_size, n_nodes, activations, L1_penalty, L2_penalty, \
            learning_rate, dropout_prob, use_batch_norm, epoch, criterion_name, criterion_params = \
                read_info_model(f"{full_directory_path}/model_informations.txt", 
                                ["num_try", "val_loss", "val_acc", "execution_time_train", 
                                 "execution_time_load_saved_model", "execution_time_use_saved_model", 
                                 "mode", "batch_size", "n_nodes", "activations", "L1_penalty", 
                                 "L2_penalty", "learning_rate", "dropout_prob", "use_batch_norm", 
                                 "num_epochs_used", "criterion_name", "criterion_params"])

            # Add the extracted data as a new line in the CSV batch writer
            writer.add_line_full(num_try, val_loss, val_acc, train_timer, mean_model_load_timer, 
                                 mean_model_timer, batch_size, n_nodes, activations, L1_penalty, 
                                 L2_penalty, learning_rate, dropout_prob, use_batch_norm, mode, 
                                 epoch, criterion_name, criterion_params)

    # Finalize the CSV writing process
    writer.close()

            
            