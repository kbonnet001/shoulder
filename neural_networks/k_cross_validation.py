from sklearn.model_selection import KFold
import os
from neural_networks.data_preparation import data_preparation_create_tensor, compute_samples
from torch.utils.data import DataLoader, random_split, Subset
from neural_networks.MuscleDataset import MuscleDataset
from neural_networks.data_tranning import *
from neural_networks.main_trainning import train_model_supervised_learning
import numpy as np

def create_dataset_from_folder_cross_val(mode, folder_name, nb_segment):
  """
  Create datasets and loaders for cross-validation: 80% for training and validation, 20% for testing.
  
  Args:
  - mode : str, mode of operation (used in data preparation).
  - folder_name : str, path to the folder containing CSV files with muscle data.
  - nb_segment : int, number of segments in the model (for data preparation).

  Returns:
  - train_val_dataset : Dataset, combined dataset for training and validation (80% of the data).
  - test_dataset : Dataset, dataset for testing (20% of the data).
  - input_size : int, the size of the input features (X).
  - output_size : int, the size of the output labels (y). (WARNING: always 1)
  """

  datasets = []
  
  # Iterate through all CSV files in the specified folder
  for filename in os.listdir(folder_name):
      if filename.endswith(".CSV"):
          file_path = os.path.join(folder_name, filename)
          print(f"Processing file: {file_path}")

          # Prepare the data from the CSV file
          all_possible_categories = list(range(nb_segment))  # Define possible categories based on segments
          X_tensor, y_tensor, _ = data_preparation_create_tensor(mode, file_path, all_possible_categories)
          dataset = MuscleDataset(X_tensor, y_tensor)

          # Split the dataset into training/validation (80%) and testing (20%)
          train_val_size, test_size = compute_samples(dataset, 0.80)
          train_val_dataset, test_dataset = random_split(dataset, [train_val_size, test_size])

          datasets.append((train_val_dataset, test_dataset))

  # Combine all train_val and test datasets from different files
  train_val_dataset = torch.utils.data.ConcatDataset([datasets[k][0] for k in range(len(datasets))])
  test_dataset = torch.utils.data.ConcatDataset([datasets[k][1] for k in range(len(datasets))])

  # Determine input and output sizes
  input_size = len(X_tensor[0])  # Number of input features
  if len(y_tensor.size()) == 2:
      output_size = y_tensor.size()[1]  # Number of output features (labels)
  else:
      output_size = 1  # Default output size if y_tensor is 1-dimensional

  return train_val_dataset, test_dataset, input_size, output_size

def new_k_loaders(batch_size, train_val_dataset, train_idx, val_idx):
    """
    Create train and validation loaders for a new k-fold split.

    Args:
    - batch_size : int, the number of samples per batch (e.g., 16, 32, 64, 128).
    - train_val_dataset : dataset, the combined dataset used for both training and validation (e.g., 80% of the data).
    - train_idx : list of int, indices for the training subset.
    - val_idx : list of int, indices for the validation subset.

    Returns:
    - train_loader : DataLoader, DataLoader for the training subset.
    - val_loader : DataLoader, DataLoader for the validation subset.
    """
    
    # Create subsets for the current fold using the provided indices
    train_subset = Subset(train_val_dataset, train_idx)
    val_subset = Subset(train_val_dataset, val_idx)

    # Create DataLoaders for the training and validation subsets
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=True)
    
    return train_loader, val_loader

def cross_validation(folder_name, Hyperparams, mode, num_folds, nb_segment):
  """
  Perform cross-validation to compute the mean test loss and accuracy of the model.
  This method provides a more reliable performance measure by reducing bias.
  
  Args:
  - folder_name : str, path to the folder containing CSV files with muscle data.
  - Hyperparams : ModelHyperparameters, contains hyperparameters for the model.
  - mode : str, mode of operation for data preparation.
  - num_folds : int, number of folds for cross-validation.
  - nb_segment : int, number of segments in the model for data preparation.
  
  Returns:
  - mean_test_loss : float, average test loss across all folds.
  - std_test_loss : float, standard deviation of test loss across all folds.
  - mean_test_acc : float, average test accuracy across all folds.
  - std_test_acc : float, standard deviation of test accuracy across all folds.
  """
  
  # Initialize KFold cross-validation
  kfold = KFold(n_splits=num_folds, shuffle=True)
  
  # Create datasets and loaders
  train_val_dataset, test_dataset, input_size, output_size = create_dataset_from_folder_cross_val(
      mode, folder_name, nb_segment)
  test_loader = DataLoader(test_dataset, batch_size=Hyperparams.batch_size, shuffle=False)

  # Lists to store results for each fold
  fold_test_losses = []
  fold_test_accs = []
  
  # Iterate over each fold
  for fold, (train_idx, val_idx) in enumerate(kfold.split(train_val_dataset)):
      print(f'Fold {fold + 1}')
      
      # Create new train and validation loaders for this fold
      train_loader, val_loader = new_k_loaders(
          Hyperparams.batch_size, train_val_dataset, train_idx, val_idx)

      # Train the model with the current fold
      model, _, _, _, _, _ = train_model_supervised_learning(
          train_loader, val_loader, test_loader, input_size, output_size, Hyperparams, 
          "", False, False)
      
      # Evaluate the model on the test set
      test_loss, test_acc, _, _ = evaluate(model, test_loader, Hyperparams.criterion)
      fold_test_losses.append(test_loss)
      fold_test_accs.append(test_acc)

  # Calculate the mean and standard deviation of test loss and accuracy
  mean_test_loss = np.mean([np.min(losses) for losses in fold_test_losses])
  std_test_loss = np.std([np.min(losses) for losses in fold_test_losses])
  mean_test_acc = np.mean([np.min(accs) for accs in fold_test_accs])
  std_test_acc = np.std([np.min(accs) for accs in fold_test_accs])

  # Print the results
  print(f'Mean Test Loss: {mean_test_loss:.6f} ± {std_test_loss:.6f}')
  print(f'Mean Test Acc: {mean_test_acc:.6f} ± {std_test_acc:.6f}')
  
  return mean_test_loss, std_test_loss, mean_test_acc, std_test_acc

def try_best_hyperparams_cross_validation(folder_name, list_simulation, num_try_cross_validation, num_folds):
  """
  Perform cross-validation with different hyperparameters to find the best set.

  This function evaluates multiple sets of hyperparameters by performing cross-validation and collects the results.

  Args:
  - folder_name : string, name of the folder containing the simulation data.
  - list_simulation : list of tuples, each containing simulation parameters and hyperparameters to be tested.
  - num_try_cross_validation : int, number of cross-validation attempts to perform for each set of hyperparameters.
  - num_folds : int, number of folds to use in cross-validation.

  Returns:
  - all_cross_val_test : list of lists, where each sublist contains the results of cross-validation for a specific set of hyperparameters.
  """
  
  all_cross_val_test = []

  for n in range(num_try_cross_validation):
      # Perform cross-validation with the current set of hyperparameters and store results
      all_cross_val_test.append([cross_validation(folder_name, list_simulation[n][1], num_folds)])
  
  print(f"Results of {num_try_cross_validation} cross validation : \n")
  for n in range(num_try_cross_validation):
      # Print the mean test loss and accuracy with their standard deviations
      print(f"n: {n} Mean Test Loss: {all_cross_val_test[n][0][0]:.6f} ± {all_cross_val_test[n][0][1]:.6f}\n", 
            f"Mean Test Acc: {all_cross_val_test[n][0][2]:.6f} ± {all_cross_val_test[n][0][3]:.6f}\n")
  
  return all_cross_val_test

