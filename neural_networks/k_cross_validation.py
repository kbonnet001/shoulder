from sklearn.model_selection import KFold
import os
from neural_networks.data_preparation import data_preparation_create_tensor, compute_samples
from torch.utils.data import DataLoader, random_split, Subset
from neural_networks.MuscleDataset import MuscleDataset
from neural_networks.data_tranning import *
from neural_networks.main_trainning import train_model_supervised_learning
import numpy as np

def create_dataset_from_folder_cross_val(mode, folder_name):
  """Create loaders : 
    80 % : train + validation
    20% : test
    Args : 
    - folder_name : string, name of the folder containing dataframe of muscles (.xlsx or .xls)
    
    Returns : 
    - train_val_loader : DataLoader, data train + val(80%)
    - test_loader : DataLoader, data testing (20%)
    - input_size : int, size of input X
    - output_size : int, size of output y (WARNING, always 1)"""

  datasets = []
    
  for filename in os.listdir(folder_name):
    if filename.endswith(".xlsx") or filename.endswith(".xls"):
        file_path = os.path.join(folder_name, filename)
        print(f"Processing file: {file_path}")

        all_possible_categories = [0,1,2,3,4,5,6,7,8,9,10,11] # number of segment in the model, look at "segment_names"
        X_tensor, y_tensor, _ = data_preparation_create_tensor(mode, file_path, all_possible_categories)
        dataset = MuscleDataset(X_tensor, y_tensor)

        train_val_size, test_size = compute_samples(dataset, 0.80)
        train_val_dataset, test_dataset = random_split(dataset, [train_val_size, test_size]) 

        datasets.append((train_val_dataset, test_dataset))

  # Merge dataset
  train_val_dataset = torch.utils.data.ConcatDataset([datasets[k][0] for k in range (len(datasets))])
  test_dataset = torch.utils.data.ConcatDataset([datasets[k][1] for k in range (len(datasets))])

  input_size = len(X_tensor[0])
  output_size = 1 # warning if y tensor change

  return train_val_dataset, test_dataset, input_size, output_size

def new_k_loaders(batch_size, train_val_dataset, train_idx, val_idx) : 
    """Create train and val loader for a new k fold
    
    Args : 
    - batch_size : int, 16, 32, 64, 128 ...
    - train_val_dataset : dataset, train and val datas (80%)
    - train_idx : [int], index train part
    - val_idx : [int], index val part 
    
    Returns : 
    - train_loader : DataLoader, data trainning 
    - val_loader : DataLoader, data validation
    """
    # Création des sous-ensembles de données pour le pli actuel
    train_subset = Subset(train_val_dataset, train_idx)
    val_subset = Subset(train_val_dataset, val_idx)

    # Création des DataLoader
    train_loader = DataLoader(train_subset, batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size, shuffle=True)
    
    return train_loader, val_loader

def cross_validation(folder_name, Hyperparams, num_folds) : 
    """Cross validation to compute mean mean_distance of the model
    Compute performance with less biais
    
    Args : 
    - folder_name : string, name of the folder containing dataframe of muscles (.xlsx or .xls)
    - Hyperparams : (ModelHyperparameters) all hyperparameters choosen by user
    - Num_folder : int, number of k folds
    
    Returns : ?
    """
    
    kfold = KFold(n_splits=num_folds, shuffle=True)
    
    train_val_dataset, test_dataset, input_size, output_size = create_dataset_from_folder_cross_val(Hyperparams.mode, folder_name) 
    test_loader = DataLoader(test_dataset, batch_size=Hyperparams.batch_size, shuffle=False)

    # Initialisation des listes pour stocker les résultats de chaque pli
    fold_test_losses = []
    fold_test_accs = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_val_dataset)):
        print(f'Fold {fold + 1}')
        
        # on fait les noueau train et val loader en fonction du k fold où on est
        train_loader, val_loader = new_k_loaders(Hyperparams.batch_size, train_val_dataset, train_idx, val_idx)

        model, _, _, _, _, _= train_model_supervised_learning(train_loader, val_loader, test_loader, input_size, output_size, Hyperparams, 
                                    "", False, False) 
        
        test_loss, test_acc = evaluate(model, test_loader, Hyperparams.criterion)
        fold_test_losses.append(test_loss)
        fold_test_accs.append(test_acc)

    # Calcul des moyennes et des écarts-types des pertes et des précisions sur les plis
    mean_test_loss = np.mean([np.min(fold) for fold in fold_test_losses])
    std_test_loss = np.std([np.min(fold) for fold in fold_test_losses])
    mean_test_acc = np.mean([np.min(fold) for fold in fold_test_accs])
    std_test_acc = np.std([np.min(fold) for fold in fold_test_accs])

    print(f'Mean Test Loss: {mean_test_loss:.6f} ± {std_test_loss:.6f}')
    print(f'Mean Test Acc: {mean_test_acc:.6f} ± {std_test_acc:.6f}')
    
    return mean_test_loss, std_test_loss,mean_test_acc, std_test_acc

def try_best_hyperparams_cross_validation(folder_name, list_simulation, num_try_cross_validation, num_folds) : 
    
    all_cross_val_test = []
    
    for n in range(num_try_cross_validation) : 
        
        all_cross_val_test.append([cross_validation(folder_name, list_simulation[n][1], num_folds)]) 
    
    print(f"Results of {num_try_cross_validation} cross validation : \n")
    for n in range (num_try_cross_validation) :
          print(f"n: {n} Mean Test Loss: {all_cross_val_test[n][0][0]:.6f} ± {all_cross_val_test[n][0][1]:.6f}\n", 
                f"Mean Test Acc: {all_cross_val_test[n][0][2]:.6f} ± {all_cross_val_test[n][0][3]:.6f}\n")
          
    return all_cross_val_test
