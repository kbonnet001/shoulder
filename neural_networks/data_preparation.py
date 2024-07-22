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
from neural_networks.other import compute_row_col, compute_num_bins
from neural_networks.Mode import Mode
import torch.nn.functional as F

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

  INPUT
  - dataset : MuscleDataset
  - train_ratio : float (0 to 1)

  OUTPUT
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
  
  INPUTS : 
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

def data_preparation_create_tensor(mode, df_data, limit, all_possible_categories):
    """
    Load data from df and create X and y tensors for PyTorch
    NOTE : normalization was deleted because x tensor are physical values (except for "muscle selected")
    
    INPUT:
    - df_data: DataFrame (.xlsx) with all data. The last column must be y
    - limit: Placeholder parameter for standardization
    - all_possible_categories: List of all possible categories for the 'index_muscle' column
    
    OUTPUT:
    - X_tensor: X tensor with all features (columns except the last one)
    - y_tensor: y tensor with the target values (last column)
    """

    # Load and standardize df
    df_muscle_datas = data_standardization(df_data, limit)
    # df_muscle_datas = df_muscle_datas.iloc[:, :]  # Adjust as necessary

    # Separate inputs from targets
    if mode == Mode.DLMT_DQ :
      X = df_muscle_datas.loc[:, 'muscle_selected':'segment_length'].values
      X = np.delete(X, (0, -2, -3, -4, -5, -6, -7), axis=1) # on met lmt mais on enleve les coordonnes de origin et insertion
    
      # Filtrer les colonnes dont les noms commencent par 'dlmt_dq_'
      selected_columns = [col for col in df_muscle_datas.columns if col.startswith('dlmt_dq_')]
      y = df_muscle_datas.loc[:, selected_columns].values
      y_labels = selected_columns
      
    else : # defaut mode = MUSCLE
      X = df_muscle_datas.loc[:, 'muscle_selected':'insertion_muscle_z'].values
      X = np.delete(X, (0, -1, -2, -3, -4, -5, -6), axis=1) # on enleve les coordonnes de origin et insertion
      
      y = df_muscle_datas.loc[:, 'segment_length'].values
      y_labels = ['segment_length']
    
    # # One-hot encoding for the 'index_muscle' column
    # encoder = OneHotEncoder(sparse_output=False, categories=[all_possible_categories])
    # index_muscle_encoded = encoder.fit_transform(X[:, 0].reshape(-1, 1))

    # # Concatenate the encoded index_muscle with the rest of the features
    # X = np.hstack((index_muscle_encoded, X[:, 1:]))

    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    return X_tensor, y_tensor, y_labels

def create_loaders_from_folder(Hyperparams, q_ranges, folder_name, with_noise = True, plot=False):
  """Create loaders : 
    80 % : train (80%) + validation (20%)
    20% : test
    INPUTS : 
    - Hyperparams : ModelHyperparameters, all hyperparameters to try, choosen by user
    - q_ranges : [q] q ranges of all q selected
    - folder_name : string, name of the folder containing dataframe of muscles (.xlsx or .xls)
    - plot : (default = False) bool, True to show datas distribution
    
    OUTPUTS : 
    - train_loader : DataLoader, data trainning (80% of 80%)
    - val_loader : DataLoader, data validation (20% of 80%)
    - test_loader : DataLoader, data testing (20%)
    - input_size : int, size of input X
    - output_size : int, size of output y (WARNING, always 1)"""
  
  filenames = sorted([filename for filename in os.listdir(folder_name)])
  # filenames[0] --> muscle
  # filenames[1] --> muscle datas ignored (could not exist...)
    
  if not (filenames[0].endswith(".xlsx") or filenames[0].endswith(".xls")):
      print("Error : File need extension .xlsx or .xls\n\
        If the file exist, maybe it's open in a window. Please close it and try again.")
      sys.exit(1)
  else : 
      file_path = os.path.join(folder_name, filenames[0])
      print(f"Processing file: {file_path}")

      all_possible_categories = [0,1,2,3,4,5,6,7,8,9,10,11] # number of segment in the model, look at "segment_names"
      X_tensor, y_tensor, y_labels = data_preparation_create_tensor(Hyperparams.mode, file_path, 0, 
                                                                    all_possible_categories)
      X_tensors=[X_tensor]
      y_tensors=[y_tensor]
      
      input_size = len(X_tensor[0])
      if Hyperparams.mode == Mode.DLMT_DQ : 
        output_size = y_tensor.size()[1]
      else : # default mode == MUSCLE
        output_size = 1 # warning if y tensor change

      if plot : 
        if os.path.exists(f"{file_path.replace(".xlsx", "")}_datas_ignored.xlsx"):
          X_tensor_ignored, y_tensor_ignored, _ = \
            data_preparation_create_tensor(Hyperparams.mode, f"{file_path.replace(".xlsx", "")}_datas_ignored.xlsx", 
                                           0, all_possible_categories)
          X_tensors.append(X_tensor_ignored)
          y_tensors.append(y_tensor_ignored)
      if plot or with_noise :    
        if os.path.exists(f"{file_path.replace(".xlsx", "")}_with_noise.xlsx"):
          X_tensor_with_noise, y_tensor_with_noise, _ = \
            data_preparation_create_tensor(Hyperparams.mode, f"{file_path.replace(".xlsx", "")}_with_noise.xlsx", 
                                           0, all_possible_categories)
          X_tensors.append(X_tensor_with_noise)
          y_tensors.append(y_tensor_with_noise)
          
        plot_datas_distribution(filenames[0],folder_name, q_ranges, X_tensors, y_tensors, y_labels)
      
      # X_tensor = F.normalize(X_tensor)  # Normalize each row (sample) to have unit norm
      
      if with_noise and os.path.exists(f"{file_path.replace(".xlsx", "")}_with_noise.xlsx"): 
        dataset = MuscleDataset(torch.cat((X_tensor, X_tensor_with_noise), dim=0), torch.cat((y_tensor, y_tensor_with_noise), dim=0))
      else : 
        dataset = MuscleDataset(X_tensor, y_tensor)

      train_val_size, test_size = compute_samples(dataset, 0.80)
      train_val_dataset, test_dataset = random_split(dataset, [train_val_size, test_size]) 

      train_size, val_size = compute_samples(train_val_dataset, 0.80)
      train_dataset, val_dataset = random_split(train_val_dataset, [train_size, val_size])
      
      train_loader = DataLoader(train_dataset, batch_size=Hyperparams.batch_size, shuffle=True)
      val_loader = DataLoader(val_dataset, batch_size=Hyperparams.batch_size, shuffle=True)
      test_loader = DataLoader(test_dataset, batch_size=Hyperparams.batch_size, shuffle=False)
      
      return train_loader, val_loader, test_loader, input_size, output_size, y_labels


def create_loaders_from_folder_group_muscle(Hyperparams, q_ranges, folder_name, plot=False):
  """
  NOT USED 
  
  Create loaders : 
    80 % : train (80%) + validation (20%)
    20% : test
    INPUTS : 
    - batch_size : int, 16, 32, 64, 128 ...
    - folder_name : string, name of the folder containing dataframe of muscles (.xlsx or .xls)
    - plot : (default = False) bool, True to show datas distribution
    
    OUTPUTS : 
    - train_loader : DataLoader, data trainning (80% of 80%)
    - val_loader : DataLoader, data validation (20% of 80%)
    - test_loader : DataLoader, data testing (20%)
    - input_size : int, size of input X
    - output_size : int, size of output y (WARNING, always 1)"""

  datasets = []
    
  for filename in os.listdir(folder_name):
    if filename.endswith(".xlsx") or filename.endswith(".xls"):
        file_path = os.path.join(folder_name, filename)
        print(f"Processing file: {file_path}")

        all_possible_categories = [0,1,2,3,4,5,6,7,8,9,10,11] # number of segment in the model, look at "segment_names"
        X_tensor, y_tensor = data_preparation_create_tensor(file_path, 0, all_possible_categories)
        dataset = MuscleDataset(X_tensor, y_tensor)

        train_val_size, test_size = compute_samples(dataset, 0.80)
        train_val_dataset, test_dataset = random_split(dataset, [train_val_size, test_size]) 

        train_size, val_size = compute_samples(train_val_dataset, 0.80)
        train_dataset, val_dataset = random_split(train_val_dataset, [train_size, val_size])

        datasets.append((train_dataset, val_dataset, test_dataset))

        if plot : 
          plot_datas_distribution(Hyperparams, filename, file_path, q_ranges, X_tensor, y_tensor)
          

  # Merge dataset
  train_dataset = torch.utils.data.ConcatDataset([datasets[k][0] for k in range (len(datasets))])
  val_dataset = torch.utils.data.ConcatDataset([datasets[k][1] for k in range (len(datasets))])
  test_dataset = torch.utils.data.ConcatDataset([datasets[k][2] for k in range (len(datasets))])

  train_loader = DataLoader(train_dataset, batch_size=Hyperparams.batch_size, shuffle=True)
  val_loader = DataLoader(val_dataset, batch_size=Hyperparams.batch_size, shuffle=True)
  test_loader = DataLoader(test_dataset, batch_size=Hyperparams.batch_size, shuffle=False)

  input_size = len(X_tensor[0])
  output_size = 1 # warning if y tensor change

  return train_loader, val_loader, test_loader, input_size, output_size

# ----------------------------------------------------------------------------------------------

def create_data_loader(mode, filename, limit, all_possible_categories) : 
  X_tensor, y_tensor, y_labels = data_preparation_create_tensor(mode, filename, limit, all_possible_categories)
  dataset = MuscleDataset(X_tensor, y_tensor)
  loader = DataLoader(dataset, 32, shuffle = False)
  return loader 

def plot_datas_distribution(filename, files_path, q_ranges, X_tensors, y_tensors, y_labels):
    """To visualise tensors distribution
    Note : This function was written in this file and not in "plot_visualisation" to avoid a circular import

    INPUT : 
    - filename : name of the excel file with datas of the muscle (good datas) 
    - files_path : file_path to save the plot
    - q_ranges : [q], all q to see distribution 
    - X_tensors : [X tensor], X tensor with all features (columns except the last one)
    - y_tensors : [y tensor], y tensor with the target values (last column) """
    
    row_fixed, col_fixed = compute_row_col(len(q_ranges) + len(y_labels), 4)
    
    fig, axs = plt.subplots(row_fixed, col_fixed, figsize=(15, 10)) 
    
    for i in range(len(q_ranges)):
        row = i // 4  
        col = i % 4   
        axs[row, col].hist([X_tensors[k][:, i] for k in range (len(X_tensors))], bins=20, alpha=0.5, stacked=True, 
                           label=["datas for learning", "datas ignored", "datas with noise"])
        axs[row, col].set_xlabel('Value')
        axs[row, col].set_ylabel('Frequency')
        axs[row, col].set_title(f'Distribution of q{i}')
        axs[row, col].legend()
    
    if len(y_labels) == 1 : 
      axs[row_fixed-1, col_fixed-1].hist([y_tensors[k] for k in range (len(y_tensors))], bins=20, alpha=0.5, stacked=True,
                                      label=["datas for learning", "datas ignored", "datas with noise"])
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
        # num_bins = int((abs(x_max) + abs(x_min)) * 1000)
    
        axs[row_j, col_j].hist(y_plot, bins=500, alpha=0.5, stacked=True, 
                            label=["datas for learning", "datas ignored", "datas with noise"])
        axs[row_j, col_j].set_xlim([x_min, x_max])
        axs[row_j, col_j].set_xlabel('Value')
        axs[row_j, col_j].set_ylabel('Frequency')
        axs[row_j, col_j].set_title(f'Distribution of {y_labels[j]}')
        axs[row_j, col_j].legend()

    fig.suptitle(f'Distribution of q and y_tensor - {filename.replace(".xlsx", "")}', fontweight='bold')
    plt.tight_layout()  
    create_and_save_plot(files_path, f"_plot_datas_distribution_{filename.replace(".xlsx", "")}")
    plt.show()
