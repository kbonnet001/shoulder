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
from neural_networks.other import *
from neural_networks.Mode import Mode
import torch.nn.functional as F
import time

def print_informations_environment() : 
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
# ---------------------------------------------------------------------------------------------

def compute_samples(dataset, train_ratio) :
  """Compute samples with a train_ratio to separate tensor

  Args
  - dataset : MuscleDataset
  - train_ratio : float (0 to 1)

  Returns
  - n_train_samples : int, number of sample for tranning
  - n_test_samples : int, number of sample for testing """

  n_samples = len(dataset)
  n_train_samples = int(n_samples * train_ratio)
  n_test_samples = n_samples - n_train_samples

  return n_train_samples, n_test_samples

def data_standardization(filename, limit = 0):
  """Delete useless lines of datafame (pov y). 
  The output filename have a "limit" of datas beetween 0.xx and 0.xx
  Simply, try to avoid normal distribution of y 
  
  Args : 
  - filename : name path of the dataframe
  - limit : (defaul = 0) int, limit off data by steps
  
  OUPUT : 
  - dataframe : copy of input dataframe with some deleted lines"""
  
  df2 = pd.read_excel(filename)
  if limit != 0 :
    df = pd.read_excel(filename)

    distribution = np.zeros(30)
    k=0
    while k <(df2.shape[0]):
      value = df.iloc[k, -1] 
      print("k = ", k)
      
      for i in range(30):
          lower_bound = 0.10 + i * 0.01
          upper_bound = 0.11 + i * 0.01
          if lower_bound <= value < upper_bound:
              if distribution[i] < limit :
                distribution[i] += 1
              else :
                df2 = df2.drop(k)
                df2 = df2.reset_index(drop=True)
              break
      k+=1

    print("Distribution:", distribution)

  return df2

# -------------------------------------------------------------------------

def get_y_and_labels(mode, df_datas, get_y = True) : 
  
  y = None

  if mode == Mode.DLMT_DQ :
    selected_columns = [col for col in df_datas.columns if col.startswith('dlmt_dq_')]
  
  elif mode == Mode.MUSCLE_DLMT_DQ : 
    # Filtrer les colonnes dont les noms commencent par 'dlmt_dq_'
    selected_columns = [col for col in df_datas.columns if col.startswith('dlmt_dq_')]
    selected_columns.insert(0, 'segment_length')
    
  elif mode == Mode.TORQUE : 
    selected_columns = ['torque']
  
  elif mode == Mode.TORQUE_MUS_DLMT_DQ : 
    selected_columns = [col for col in df_datas.columns if col.startswith('dlmt_dq_')]
    selected_columns.insert(0, 'segment_length')
    selected_columns.insert(len(selected_columns), 'torque')
    
  else : # defaut mode = MUSCLE
    selected_columns = ['segment_length']
  
  y_labels = selected_columns
  if get_y : 
    y = df_datas.loc[:, selected_columns].values
    
  return y, y_labels
  

def data_preparation_create_tensor(mode, file_path_df, all_possible_categories):
    """
    Load data from df and create X and y tensors for PyTorch
    NOTE : normalization was deleted because x tensor are physical values (except for "muscle selected")
    
    Args:
    - file_path_df : path for excel file with datas
    - limit: Placeholder parameter for standardization
    - all_possible_categories: List of all possible categories for the 'index_muscle' column
    
    Returns:
    - X_tensor: X tensor with all features (columns except the last one)
    - y_tensor: y tensor with the target values (last column)
    """
    # Load and standardize df
    # df_datas = data_standardization(df_data, limit)
    df_datas = pd.read_excel(file_path_df)
    
    selected_columns_q = [col for col in df_datas.columns if col.startswith('q_')]
    X = df_datas.loc[:, selected_columns_q].values # q only

    y, y_labels = get_y_and_labels(mode, df_datas, get_y = True)
    
    # # One-hot encoding for the 'index_muscle' column
    # encoder = OneHotEncoder(sparse_output=False, categories=[all_possible_categories])
    # index_muscle_encoded = encoder.fit_transform(X[:, 0].reshape(-1, 1))

    # # Concatenate the encoded index_muscle with the rest of the features
    # X = np.hstack((index_muscle_encoded, X[:, 1:]))

    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    return X_tensor, y_tensor, y_labels

def create_loaders_from_folder(Hyperparams, mode, nbQ, num_datas_for_dataset, folder_name, muscle_name, with_noise = True, plot=False):
  """Create loaders : 
    80 % : train (80%) + validation (20%)
    20% : test
    Args : 
    - Hyperparams : ModelHyperparameters, all hyperparameters to try, choosen by user
    - q_ranges : [q] q ranges of all q selected
    - folder_name : string, name of the folder containing dataframe of muscles (.xlsx or .xls)
    - with_noise : (default = True), bool, true to put datas with noise in dataset for learning
    - plot : (default = False) bool, True to show datas distribution
    
    Returns : 
    - train_loader : DataLoader, data trainning (80% of 80%)
    - val_loader : DataLoader, data validation (20% of 80%)
    - test_loader : DataLoader, data testing (20%)
    - input_size : int, size of input X
    - output_size : int, size of output y (WARNING, always 1)"""
    
  file_path_df = os.path.join(folder_name, f"{muscle_name}.xlsx")
    
  if not os.path.exists(file_path_df):
      error = "Error : File need extension .xlsx or .xls\n\
        If the file exist, maybe it's open in a window. Please close it and try again."
      sys.exit(error)
  else : 
      print(f"Processing file: {file_path_df}")

      all_possible_categories = [0,1,2,3,4,5,6,7,8,9,10,11] # number of segment in the model, look at "segment_names"
      X_tensor, y_tensor, y_labels = data_preparation_create_tensor(mode, file_path_df, 
                                                                    all_possible_categories)
      X_tensors=[X_tensor]
      y_tensors=[y_tensor]
      
      input_size = len(X_tensor[0])
      if len(y_tensor.size()) == 2 :
        output_size = y_tensor.size()[1]
      else : 
        output_size = 1

      # datas with noise   
      if with_noise :
        if os.path.exists(f"{file_path_df.replace(".xlsx", "_with_noise.xlsx")}"):
          X_tensor_with_noise, y_tensor_with_noise, _ = \
            data_preparation_create_tensor(mode, f"{file_path_df.replace(".xlsx", "_with_noise.xlsx")}", 
                                           all_possible_categories)
      # if plot
      if plot : 
        graph_labels = ["datas for learning"]
        
        if with_noise : 
          X_tensors.append(X_tensor_with_noise)
          y_tensors.append(y_tensor_with_noise)
          graph_labels.append("datas with noise")
        
        if os.path.exists(f"{file_path_df.replace(".xlsx", "_datas_ignored.xlsx")}"):
          X_tensor_ignored, y_tensor_ignored, _ = \
            data_preparation_create_tensor(mode, f"{file_path_df.replace(".xlsx", "_datas_ignored.xlsx")}", 
                                           all_possible_categories)
          X_tensors.append(X_tensor_ignored)
          y_tensors.append(y_tensor_ignored)
          graph_labels.append("datas ignored")
 
        plot_datas_distribution(mode, muscle_name,folder_name, nbQ, X_tensors, y_tensors, y_labels, graph_labels)
      
      # Normalize each row (sample) to have unit norm, avoid it if datas have physical unit
      # X_tensor = F.normalize(X_tensor)  
      
      # Selecte datas to put in the dataset
      if with_noise and os.path.exists(f"{file_path_df.replace(".xlsx", "_with_noise.xlsx")}"): 
        dataset = MuscleDataset(torch.cat((X_tensor, X_tensor_with_noise), dim=0), torch.cat((y_tensor, y_tensor_with_noise), dim=0))
      else : 
        dataset = MuscleDataset(X_tensor, y_tensor)

      # Selecte the good number of datas in dataset (user's choice)
      if len(dataset) > num_datas_for_dataset : 
        dataset.remove_random_items(len(dataset) - num_datas_for_dataset)
      else : 
        print("\nAll dataset will be use, len(dataset) = ", len(dataset))
        time.sleep(10)
      
      train_val_size, test_size = compute_samples(dataset, 0.80)
      train_val_dataset, test_dataset = random_split(dataset, [train_val_size, test_size]) 

      train_size, val_size = compute_samples(train_val_dataset, 0.80)
      train_dataset, val_dataset = random_split(train_val_dataset, [train_size, val_size])
      
      train_loader = DataLoader(train_dataset, batch_size=Hyperparams.batch_size, shuffle=True)
      val_loader = DataLoader(val_dataset, batch_size=Hyperparams.batch_size, shuffle=True)
      test_loader = DataLoader(test_dataset, batch_size=Hyperparams.batch_size, shuffle=False)
      
      return train_loader, val_loader, test_loader, input_size, output_size, y_labels


# def create_loaders_from_folder_group_muscle(Hyperparams, nbQ, folder_name, plot=False):
#   """
#   NOT USED 
  
#   Create loaders : 
#     80 % : train (80%) + validation (20%)
#     20% : test
#     Args : 
#     - batch_size : int, 16, 32, 64, 128 ...
#     - folder_name : string, name of the folder containing dataframe of muscles (.xlsx or .xls)
#     - plot : (default = False) bool, True to show datas distribution
    
#     Returns : 
#     - train_loader : DataLoader, data trainning (80% of 80%)
#     - val_loader : DataLoader, data validation (20% of 80%)
#     - test_loader : DataLoader, data testing (20%)
#     - input_size : int, size of input X
#     - output_size : int, size of output y (WARNING, always 1)"""

#   datasets = []
    
#   for filename in os.listdir(folder_name):
#     if filename.endswith(".xlsx") or filename.endswith(".xls"):
#         file_path = os.path.join(folder_name, filename)
#         print(f"Processing file: {file_path}")

#         all_possible_categories = [0,1,2,3,4,5,6,7,8,9,10,11] # number of segment in the model, look at "segment_names"
#         X_tensor, y_tensor = data_preparation_create_tensor(file_path, all_possible_categories)
#         dataset = MuscleDataset(X_tensor, y_tensor)

#         train_val_size, test_size = compute_samples(dataset, 0.80)
#         train_val_dataset, test_dataset = random_split(dataset, [train_val_size, test_size]) 

#         train_size, val_size = compute_samples(train_val_dataset, 0.80)
#         train_dataset, val_dataset = random_split(train_val_dataset, [train_size, val_size])

#         datasets.append((train_dataset, val_dataset, test_dataset))

#         if plot : 
#           plot_datas_distribution(Hyperparams, filename, file_path, nbQ, X_tensor, y_tensor)
          

#   # Merge dataset
#   train_dataset = torch.utils.data.ConcatDataset([datasets[k][0] for k in range (len(datasets))])
#   val_dataset = torch.utils.data.ConcatDataset([datasets[k][1] for k in range (len(datasets))])
#   test_dataset = torch.utils.data.ConcatDataset([datasets[k][2] for k in range (len(datasets))])

#   train_loader = DataLoader(train_dataset, batch_size=Hyperparams.batch_size, shuffle=True)
#   val_loader = DataLoader(val_dataset, batch_size=Hyperparams.batch_size, shuffle=True)
#   test_loader = DataLoader(test_dataset, batch_size=Hyperparams.batch_size, shuffle=False)

#   input_size = len(X_tensor[0])
#   output_size = 1 # warning if y tensor change

#   return train_loader, val_loader, test_loader, input_size, output_size

# ----------------------------------------------------------------------------------------------

def create_data_loader(mode, filename, all_possible_categories) : 
  X_tensor, y_tensor, y_labels = data_preparation_create_tensor(mode, filename, all_possible_categories)
  dataset = MuscleDataset(X_tensor, y_tensor)
  loader = DataLoader(dataset, 32, shuffle = False)
  return loader, y_labels

def dataloader_to_tensor(loader):
    # Listes pour stocker les données et les labels
    all_data = []
    all_labels = []
    
    for data, labels in loader:
        all_data.append(data)
        all_labels.append(labels)
    
    # Concaténer toutes les batchs en un seul tensor
    all_data_tensor = torch.cat(all_data)
    all_labels_tensor = torch.cat(all_labels)
    
    return all_data_tensor, all_labels_tensor


def plot_datas_distribution(mode, muscle_name, files_path, nbQ, X_tensors, y_tensors, y_labels, graph_labels):
    """To visualise tensors distribution
    Note : This function was written in this file and not in "plot_visualisation" to avoid a circular import

    Args : 
    - muscle_name : name of the excel file with datas of the muscle (good datas) 
    - files_path : file_path to save the plot
    - nbQ : number of q in model file biorbd
    - X_tensors : [X tensor], X tensor with all features (columns except the last one)
    - y_tensors : [y tensor], y tensor with the target values (last column) """
    
    row_fixed, col_fixed = compute_row_col(nbQ + len(y_labels), 4)
    
    fig, axs = plt.subplots(row_fixed, col_fixed, figsize=(15, 10)) 
    
    for i in range(nbQ):
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
      axs[row_fixed-1, col_fixed-1].hist([y_tensors[k].squeeze(1) for k in range (len(y_tensors))], bins=20, alpha=0.5, stacked=True,
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
        # num_bins = sturges_rule(y_plot[0])
        # num_bins = rice_rule(y_plot[0])
        # num_bins = scott_rule(y_plot[0].numpy())
        
        # if num_bins < 100 : 
        #   num_bins = num_bins // 2
        # if num_bins > 100 : 
        #   num_bins = num_bins * 2
        # num_bins = int((abs(x_max) + abs(x_min)) * 1000)
    
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
