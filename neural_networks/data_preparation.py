import platform
import numpy as np
import pandas as pd
import torch
import os
from torch.utils.data import DataLoader, random_split
import sklearn
from neural_networks.MuscleDataset import MuscleDataset
from neural_networks.plot_visualisation import plot_datas_distribution

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

def data_preparation_create_tensor(df_data, limit) :

  """ Load datas of df and create X and y tensors PyTorch
  NOTE : normalization was deleted because x tensor are physical values (except for "muscle selected")
  
  INPUT
  - df_data : data frame (.xlsx) with all datas. The last column must be y
  
  OUTPUT
  - X_tensor : X tensor with all features (columns except the last one)
  - y_tensor : y tensor with the target values (last column) """

  # Load df
  df_muscle_datas = data_standardization(df_data, limit)
  df_muscle_datas = df_muscle_datas.iloc[:, 1:] # on enleve la premiere colonne

  # Separate inputs from targets
  X = df_muscle_datas.iloc[:, 0:-1].values
  y = df_muscle_datas.iloc[:, -1].values

  # Normalisation
  # scaler = StandardScaler()
  # X_scaled = scaler.fit_transform(X)

  # # Convertir en tensors PyTorch
  # X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
  
  X_tensor = torch.tensor(X, dtype=torch.float32)
  y_tensor = torch.tensor(y, dtype=torch.float32)

  return X_tensor, y_tensor

def create_loaders_from_folder(batch_size, folder_name, plot=False):
  """Create loaders : 
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

        X_tensor, y_tensor = data_preparation_create_tensor(filename, 0)
        dataset = MuscleDataset(X_tensor, y_tensor)

        train_val_size, test_size = compute_samples(dataset, 0.80)
        train_val_dataset, test_dataset = random_split(dataset, [train_val_size, test_size]) 

        train_size, val_size = compute_samples(train_val_dataset, 0.80)
        train_dataset, val_dataset = random_split(train_val_dataset, [train_size, val_size])

        datasets.append((train_dataset, val_dataset, test_dataset))

        if plot : 
          plot_datas_distribution(X_tensor, y_tensor)

  # Merge dataset
  train_dataset = torch.utils.data.ConcatDataset([datasets[k][0] for k in range (len(datasets))])
  val_dataset = torch.utils.data.ConcatDataset([datasets[k][1] for k in range (len(datasets))])
  test_dataset = torch.utils.data.ConcatDataset([datasets[k][2] for k in range (len(datasets))])

  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
  val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
  test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

  input_size = len(X_tensor[0])
  output_size = 1 # warning if y tensor change

  return train_loader, val_loader, test_loader, input_size, output_size

