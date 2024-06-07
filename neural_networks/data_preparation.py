import platform
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch import nn, optim

import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

import sklearn
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt

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
  """Delete useless lines of datafame. 
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
  #
  # INPUT
  # - df_data : data frame (.xlsx) with all datas. The last column must be y
  #
  # OUTPUT
  # - X_tensor : X tensor with all features (columns except the last one)
  # - y_tensor : y tensor with the target values (last column) """

  # Load df
  df_muscle_datas = data_standardization(df_data, limit)
  df_muscle_datas = df_muscle_datas.iloc[:, 1:] # on enleve les 2 premiere colonne

  # Separate inputs from targets
  X = df_muscle_datas.iloc[:, 0:-1].values
  y = df_muscle_datas.iloc[:, -1].values

  # Normalisation
  scaler = StandardScaler()
  X_scaled = scaler.fit_transform(X)

  # Convertir en tensors PyTorch
  X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
  y_tensor = torch.tensor(y, dtype=torch.float32)

  return X_tensor, y_tensor

def plot_datas_distribution(X_tensor, y_tensor):
    """To visualise tensors distribution

    INPUT : 
    - X_tensor : X tensor with all features (columns except the last one)
    - y_tensor : y tensor with the target values (last column) """
    
    fig, axs = plt.subplots(2, 3, figsize=(15, 10)) 
    
    for i in range(4):
        row = i // 3  
        col = i % 3   
        axs[row, col].hist(X_tensor[:, i], bins=20, alpha=0.5)
        axs[row, col].set_xlabel('Value')
        axs[row, col].set_ylabel('Frequency')
        axs[row, col].set_title(f'Distribution of q{i+1}')

    axs[1, 2].hist(y_tensor, bins=20, alpha=0.5)  
    axs[1, 2].set_xlabel('Value')
    axs[1, 2].set_ylabel('Frequency')
    axs[1, 2].set_title('Distribution of muscle length')

    plt.tight_layout()  
    plt.show()

