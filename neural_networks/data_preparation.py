import platform
import numpy as np
import pandas as pd
import matplotlib as plt
import matplotlib.pyplot as plt
import torch
import os
import sys
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import DataLoader, random_split
import sklearn
from neural_networks.MuscleDataset import MuscleDataset
from neural_networks.file_directory_operations import create_and_save_plot
from neural_networks.muscle_plotting_utils import *
from neural_networks.Mode import Mode
import torch.nn.functional as F
import time

def print_informations_environment() : 
  """ Print information about the current Python environment and hardware setup. """
  
  # Print environment info
  print(f"Python version: {platform.python_version()}")
  print(f"NumPy version: {np.__version__}")
  print(f"scikit-learn version: {sklearn.__version__}")
  print(f"PyTorch version: {torch.__version__}")

  # PyTorch device configuration
  if torch.cuda.is_available():
      device = torch.device("cuda")
      print(f"CUDA GPU {torch.cuda.get_device_name(0)} found :)")
  # Performance issues exist with MPS backend
  # elif torch.backends.mps.is_available():
  #     device = torch.device("mps")
  #     print("MPS GPU found :)")
  else:
      device = torch.device("cpu")
      print("No GPU found, using CPU instead")
      
  # Return the device for potential further use
  return device
# ---------------------------------------------------------------------------------------------

def compute_samples(dataset, train_ratio) :
  """
  Compute the number of samples for training and testing based on the given ratio.

  Args:
  - dataset (Dataset): The dataset from which to compute the sample sizes.
  - train_ratio (float): The ratio of the dataset to use for training (between 0 and 1).

  Returns:
  - n_train_samples (int): Number of samples allocated for training.
  - n_test_samples (int): Number of samples allocated for testing.
  """

  n_samples = len(dataset)
  n_train_samples = int(n_samples * train_ratio)
  n_test_samples = n_samples - n_train_samples

  return n_train_samples, n_test_samples

# -------------------------------------------------------------------------

def get_x(mode, df_datas, get_origin_and_insertion = False) : 
    """
    Extract feature columns from a DataFrame based on the given mode.

    Args:
    - mode (Mode): The mode specifying which columns to select from the DataFrame.
    - df_datas (pd.DataFrame): The DataFrame containing the data.
    - get_origin_and_insertion (bool, optional): Whether to include columns related to muscle origin and insertion. 
      Defaults to False.

    Returns:
    - numpy.ndarray: Array of selected features from the DataFrame.
    """

    # Select columns based on the mode
    if mode in [Mode.DLMT_DQ, Mode.MUSCLE_DLMT_DQ, Mode.MUSCLE]:
        # For these modes, select columns starting with 'q_'
        selected_columns = [col for col in df_datas.columns if col.startswith('q_')]
    
    elif mode in [Mode.TORQUE, Mode.TORQUE_MUS_DLMT_DQ, Mode.DLMT_DQ_FM, Mode.FORCE, Mode.DLMT_DQ_F_TORQUE]:
        # For these modes, select columns starting with 'q_', 'qdot_', and 'alpha'
        selected_columns_q = [col for col in df_datas.columns if col.startswith('q_')]
        selected_columns_qdot = [col for col in df_datas.columns if col.startswith('qdot_')]
        selected_columns = selected_columns_q + selected_columns_qdot + ['alpha']
    
    else:
        # Raise an error if the mode is not recognized
        raise ValueError(f"Invalid mode: {mode}. The mode does not exist or is not supported.")
    
    # Add origin and insertion columns if requested
    if get_origin_and_insertion:
        selected_columns += ['origin_muscle_y', 'origin_muscle_z', 'origin_muscle_x', 'insertion_x', 'insertion_y', 'insertion_z']
    
    # Extract the selected columns from the DataFrame
    x = df_datas.loc[:, selected_columns].values
    
    return x

def get_y_and_labels(mode, df_datas, get_y = True) : 
  """
  Extract target columns and their labels from a DataFrame based on the given mode.

  Args:
  - mode (Mode): The mode specifying which target columns to select from the DataFrame.
  - df_datas (pd.DataFrame): The DataFrame containing the data.
  - get_y (bool, optional): Whether to extract the target values. Defaults to True.

  Returns:
  - y (numpy.ndarray or None): Array of target values from the DataFrame, or None if get_y is False.
  - y_labels (list of str): List of selected column names for the target.
  """
  
  y = None
  
  # Determine which columns to select based on the mode
  if mode == Mode.MUSCLE:
    selected_columns = ['segment_length']
    
  elif mode == Mode.DLMT_DQ :
    selected_columns = [col for col in df_datas.columns if col.startswith('dlmt_dq_')]
  
  elif mode == Mode.MUSCLE_DLMT_DQ : 
    selected_columns = [col for col in df_datas.columns if col.startswith('dlmt_dq_')]
    selected_columns.insert(0, 'segment_length')
    
  elif mode == Mode.TORQUE : 
    selected_columns = ['torque']
  
  elif mode == Mode.TORQUE_MUS_DLMT_DQ : 
    selected_columns = [col for col in df_datas.columns if col.startswith('dlmt_dq_')]
    selected_columns.insert(0, 'segment_length')
    selected_columns.insert(len(selected_columns), 'torque')
    
  elif mode == Mode.DLMT_DQ_FM : 
    selected_columns = [col for col in df_datas.columns if col.startswith('dlmt_dq_')]
    selected_columns.insert(0, 'muscle_force')
  
  elif mode == Mode.FORCE: 
    selected_columns = ['muscle_force']
  
  elif mode == Mode.DLMT_DQ_F_TORQUE : 
    selected_columns = [col for col in df_datas.columns if col.startswith('dlmt_dq_')]
    selected_columns.append('muscle_force')
    selected_columns.append('torque')
    
  else : # mode doesn't exist
    raise ValueError(f"Invalid mode: {mode}. The mode does not exist or is not supported.")
  
  y_labels = selected_columns
  if get_y : 
    y = df_datas.loc[:, selected_columns].values
    
  return y, y_labels
  
def data_preparation_create_tensor(mode, file_path_df, all_possible_categories):
    """
    Load data from a CSV file and create X and y tensors for PyTorch.
    Note: Normalization was removed because x tensor contains physical values (except for "muscle selected").

    Args:
    - mode (Mode): The mode specifying how to extract features and targets.
    - file_path_df (str): Path to the CSV file containing the data.
    - all_possible_categories (list of str): List of all possible categories for the 'index_muscle' column.

    Returns:
    - x_tensor (torch.Tensor): Tensor containing all features (columns except the last one).
    - y_tensor (torch.Tensor): Tensor containing the target values (last column).
    - y_labels (list of str): List of column names used for the target values.
    """
    # Load data from CSV file
    df_datas = pd.read_csv(file_path_df)
    
    # Extract features (X) and targets (y)
    x = get_x(mode, df_datas, get_origin_and_insertion = False)
    y, y_labels = get_y_and_labels(mode, df_datas, get_y = True)
    
    # Optional: One-hot encoding for the 'index_muscle' column
    # Uncomment if needed for specific modes
    # encoder = OneHotEncoder(sparse_output=False, categories=[all_possible_categories])
    # index_muscle_encoded = encoder.fit_transform(X[:, 0].reshape(-1, 1))
    # X = np.hstack((index_muscle_encoded, X[:, 1:]))

    # Convert to PyTorch tensors
    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    return x_tensor, y_tensor, y_labels

def create_loaders_from_folder(Hyperparams, mode, nb_q, nb_segment, num_datas_for_dataset, folder_name, muscle_name, 
                               with_noise = True, plot=False):
  """
  Create data loaders: 
  80%: train (80%) + validation (20%)
  20%: test
  
  Args: 
  - Hyperparams (ModelHyperparameters): All hyperparameters selected by the user.
  - mode (Mode): Mode specifying the type of data extraction and processing.
  - nb_q (int): Number of degrees of freedom.
  - nb_segment: int, number of segment in the biorbd model.
  - num_datas_for_dataset (int): Number of data samples to include in the dataset.
  - folder_name (str): Name of the folder containing the muscle dataframes (.csv).
  - muscle_name (str): Name of the muscle for which to load data.
  - with_noise (bool): Whether to include data with noise in the dataset. Default is True.
  - plot (bool): Whether to show data distribution plots. Default is False.
  
  Returns: 
  - train_loader (DataLoader): DataLoader for training data (80% of 80%).
  - val_loader (DataLoader): DataLoader for validation data (20% of 80%).
  - test_loader (DataLoader): DataLoader for testing data (20%).
  - input_size (int): Size of input X.
  - output_size (int): Size of output y (always 1).
  - y_labels (list): Labels for the output tensor.
  """
    
  file_path_df = os.path.join(folder_name, f"{muscle_name}.csv")
    
  if not os.path.exists(file_path_df):
      error = "Error : File need extension .csv \n\
        If the file exist, maybe it's open in a window. Please close it and try again."
      sys.exit(error)
  else : 
      print(f"Processing file: {file_path_df}")

      all_possible_categories = list(range(nb_segment)) # number of segment in the model, look at "segment_names"
      X_tensor, y_tensor, y_labels = data_preparation_create_tensor(mode, file_path_df, 
                                                                    all_possible_categories)
      X_tensors=[X_tensor]
      y_tensors=[y_tensor]
      
      input_size = len(X_tensor[0])
      if len(y_tensor.size()) == 2 :
        output_size = y_tensor.size()[1]
      else : 
        output_size = 1

      # Load data with noise if available
      if with_noise :
        if os.path.exists(f"{file_path_df.replace(".csv", "_with_noise.csv")}"):
          X_tensor_with_noise, y_tensor_with_noise, _ = \
            data_preparation_create_tensor(mode, f"{file_path_df.replace(".csv", "_with_noise.csv")}", 
                                           all_possible_categories)
      # Plot data distribution if requested
      if plot : 
        graph_labels = ["datas for learning"]
        
        if with_noise : 
          X_tensors.append(X_tensor_with_noise)
          y_tensors.append(y_tensor_with_noise)
          graph_labels.append("datas with noise")
        
        if os.path.exists(f"{file_path_df.replace(".csv", "_datas_ignored.csv")}"):
          X_tensor_ignored, y_tensor_ignored, _ = \
            data_preparation_create_tensor(mode, f"{file_path_df.replace(".csv", "_datas_ignored.csv")}", 
                                           all_possible_categories)
          X_tensors.append(X_tensor_ignored)
          y_tensors.append(y_tensor_ignored)
          graph_labels.append("datas ignored")
 
        plot_datas_distribution(mode, muscle_name,folder_name, nb_q, X_tensors, y_tensors, y_labels, graph_labels)
      
        # Normalize each row (sample) to have unit norm; avoid if data have physical units
        # X_tensor = F.normalize(X_tensor)
      
      # Selecte datas to put in the dataset
      if with_noise and os.path.exists(f"{file_path_df.replace(".csv", "_with_noise.csv")}"): 
        dataset = MuscleDataset(torch.cat((X_tensor, X_tensor_with_noise), dim=0), 
                                torch.cat((y_tensor, y_tensor_with_noise), dim=0))
      else : 
        dataset = MuscleDataset(X_tensor, y_tensor)

      # Selecte the good number of datas in dataset (user's choice)
      if len(dataset) > num_datas_for_dataset : 
        dataset.remove_random_items(len(dataset) - num_datas_for_dataset)
      else : 
        print("\nAll dataset will be use, len(dataset) = ", len(dataset))
        time.sleep(5)
      
      # Split dataset into train+val and test
      train_val_size, test_size = compute_samples(dataset, 0.80)
      train_val_dataset, test_dataset = random_split(dataset, [train_val_size, test_size])

      # Split train+val into train and val
      train_size, val_size = compute_samples(train_val_dataset, 0.80)
      train_dataset, val_dataset = random_split(train_val_dataset, [train_size, val_size])
      
      # Create data loaders
      train_loader = DataLoader(train_dataset, batch_size=Hyperparams.batch_size, shuffle=True)
      val_loader = DataLoader(val_dataset, batch_size=Hyperparams.batch_size, shuffle=True)
      test_loader = DataLoader(test_dataset, batch_size=Hyperparams.batch_size, shuffle=False)
      
      return train_loader, val_loader, test_loader, input_size, output_size, y_labels

# ----------------------------------------------------------------------------------------------

def create_data_loader(mode, batch_size, filename, all_possible_categories):
    """
    Create a data loader from a CSV file.

    Args:
    - mode (Mode): Mode specifying the type of data extraction and processing.
    - filename (str): Path to the CSV file containing the data.
    - all_possible_categories (list): List of all possible categories for the 'index_muscle' column.

    Returns:
    - loader (DataLoader): PyTorch DataLoader for the dataset.
    - y_labels (list): Labels for the output tensor.
    """
    # Prepare tensors from the data file
    X_tensor, y_tensor, y_labels = data_preparation_create_tensor(mode, filename, all_possible_categories)
    
    # Create dataset and data loader
    dataset = MuscleDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    return loader, y_labels

def dataloader_to_tensor(loader):
    """
    Convert all batches in a DataLoader to a single tensor for data and labels.

    Args:
    - loader (DataLoader): PyTorch DataLoader containing the data and labels.

    Returns:
    - all_data_tensor (Tensor): Concatenated tensor of all data samples.
    - all_labels_tensor (Tensor): Concatenated tensor of all labels.
    """
    # Lists to store data and labels from all batches
    all_data = []
    all_labels = []
    
    # Iterate through the DataLoader and collect data and labels
    for data, labels in loader:
        all_data.append(data)
        all_labels.append(labels)
    
    # Concatenate all batches into a single tensor
    all_data_tensor = torch.cat(all_data)
    all_labels_tensor = torch.cat(all_labels)
    
    return all_data_tensor, all_labels_tensor

def plot_datas_distribution(mode, muscle_name, files_path, nb_q, X_tensors, y_tensors, y_labels, graph_labels):
    """
    Visualize tensors distribution.

    Args:
    - mode: Mode, The mode of operation, indicating the type of data being processed.
    - muscle_name: string, The name of the muscle for which data is being plotted.
    - files_path: string, Path to save the plots.
    - nb_q: int, Number of 'q' variables in the model.
    - X_tensors: List of X tensors with features.
    - y_tensors: List of y tensors with target values.
    - y_labels: list of string, Labels for the y variables.
    - graph_labels: Labels for different datasets in the plot.
    """
    
    row_fixed, col_fixed = compute_row_col(nb_q + len(y_labels), 4)
    
    fig, axs = plt.subplots(row_fixed, col_fixed, figsize=(15, 10)) 
    
    for i in range(nb_q):
        row = i // 4  
        col = i % 4   
        axs[row, col].hist([X_tensors[k][:, i] for k in range (len(X_tensors))], bins=20, alpha=0.5, stacked=True, 
                           label=graph_labels)
        axs[row, col].set_xlabel('Value')
        axs[row, col].set_ylabel('Frequency')
        axs[row, col].set_title(f'Distribution of q{i}')
        axs[row, col].legend()
    
    if len(y_labels) == 1 : 
      # y_tensors = y_tensors.squeeze(1)
      axs[row_fixed-1, col_fixed-1].hist([y_tensors[k].squeeze(1) for k in range (len(y_tensors))], bins=20, alpha=0.5, 
                                         stacked=True,
                                      label=graph_labels)
      axs[row_fixed-1, col_fixed-1].set_xlabel('Value')
      axs[row_fixed-1, col_fixed-1].set_ylabel('Frequency')
      axs[row_fixed-1, col_fixed-1].set_title(f'Distribution of {y_labels}')
      axs[row_fixed-1, col_fixed-1].legend()
    
    else : 
      for j in range(len(y_labels)):
        row_j = (j+i+1) // 4  
        col_j = (j+i+1) % 4   
        
        y_plot = [y_tensors[k][:,j] for k in range (len(y_tensors))]
        x_min = min(y_plot[0])
        x_max = max(y_plot[0])
        num_bins = compute_num_bins(y_plot[0], x_max, x_min)
    
        axs[row_j, col_j].hist(y_plot, bins=num_bins, alpha=0.5, stacked=True, 
                            label=graph_labels)
        axs[row_j, col_j].set_xlim([x_min, x_max])
        axs[row_j, col_j].set_xlabel('Value')
        axs[row_j, col_j].set_ylabel('Frequency')
        axs[row_j, col_j].set_title(f'Distribution of {y_labels[j]}')
        axs[row_j, col_j].legend()

    fig.suptitle(f'Distribution of q and y_tensor - {muscle_name}', fontweight='bold')
    plt.tight_layout()  
    create_and_save_plot(files_path, f"_plot_datas_distribution_{muscle_name}_{str(mode).split(".")[-1]}")
    plt.show()
