from neural_networks.data_preparation import create_loaders_from_folder
from neural_networks.data_tranning import train_model_supervised_learning
from neural_networks.activation_functions import *
from torch.utils.data import DataLoader
from neural_networks.EarlyStopping import EarlyStopping
import os
from neural_networks.save_model import *
from neural_networks.plot_visualisation import *
from itertools import product
from neural_networks.Loss import *
from neural_networks.ModelHyperparameters import ModelHyperparameters
from neural_networks.file_directory_operations import create_directory, save_text_to_file
import time

def compute_time_testing_hyperparams(Hyperparams, time_per_configuration_secondes = 60) : 
    """Compute an estimation of execution code. 
    
    INPUTS : 
    - Hyperparams : ModelHyperparameters, all hyperparameters to try, choosen by user 
    - time_per_configuration_secondes : (default 60), positive float in second for train-eval one model
    
    OUTPUT : 
    - total_time_estimated_secondes : time estimation in secondes
    - total_time_estimated_minutes : time estimation in minutes
    - total_time_estimated_hours : time estimation in hours
    """
    
    #dans cette situation, hyperparams c'est des listes de trucs aue l'on veut tester
    n_combinations = (len(Hyperparams.n_layers) * len(Hyperparams.n_nodes) * len(Hyperparams.activations) *
                      len(Hyperparams.L1_penalty) * len(Hyperparams.L2_penalty) * len(Hyperparams.learning_rate) * 
                      len(Hyperparams.dropout_prob) * sum(len(list(product(*params.values()))) for name, params in Hyperparams.criterion))


    total_time_estimated_secondes = n_combinations * time_per_configuration_secondes
    total_time_estimated_minutes = n_combinations * time_per_configuration_secondes / 60
    total_time_estimated_hours = n_combinations * time_per_configuration_secondes / 3600
    
    return total_time_estimated_secondes, total_time_estimated_minutes, total_time_estimated_hours

def main_superised_learning(Hyperparams, nbQ, num_datas_for_dataset, folder_name, muscle_name, retrain, file_path, with_noise, plot_preparation, plot, save) : 
    
    """Main fonction for prepare, train-val-test and save a model 
    
    - Hyperparams : (ModelHyperparameters) all hyperparameters choosen by user
    PLEASE, look at examples below
    - folder_name : string, path/name of the folder contained all excel data file of muscles (one for each muscle)
    - retrain : bool, True to train the model again
    - file_path : string, name of the model will be save after tranning
    - with_noise : (default = True), bool, true to put datas with noise in dataset for learning
    - plot_preparation : bool, True to show distribution of all datas preparation
    
    ---------------------------
    Examples "single syntaxe" :
    ---------------------------
    To avoid bug, please pay attention to syntax, there is some differences with "find_best_hyperparameters"
        
    n_layers = 1 or n_layers = 2
    n_nodes = [12] or n_nodes = [12, 10]
    activations = [nn.GELU()] or activations = [nn.GELU(), nn.GELU()]
    activation_names = ["GELU"] or activation_names = ["GELU", "GELU"]
    L_penalty = 0.01 or 0.001
    learning_rate = 1e-3 or 1e-4

    criterion = ModifiedHuberLoss(delta=0.2, factor=1.0)
    p_dropout = 0.2 or 0.5"""
    
    # Create a folder for save plots
    create_directory(f"{folder_name}/{muscle_name}/_Model") # Muscle/Model
    
    train_loader, val_loader, test_loader, input_size, output_size, y_labels \
    = create_loaders_from_folder(Hyperparams, nbQ, num_datas_for_dataset, f"{folder_name}/{muscle_name}", 
                                 muscle_name, with_noise, plot_preparation)
    
    # train_model if retrain == True or if none file_path already exist
    if retrain or os.path.exists(f"{folder_name}/{muscle_name}/_Model/{file_path}") == False: 
        
        _, _, _, _ = train_model_supervised_learning(train_loader, val_loader, test_loader, input_size, output_size, 
                                                  Hyperparams, f"{folder_name}/{muscle_name}/_Model/{file_path}", 
                                                  plot, save)
        
    visualize_prediction(Hyperparams.mode, nbQ, y_labels, train_loader, val_loader, test_loader, 
                         f"{folder_name}/{muscle_name}/_Model/{file_path}", 
                         f"{folder_name}/{muscle_name}/plot_all_q_variation_")
    
    
def find_best_hyperparameters(Hyperparams, nbQ, num_datas_for_dataset, folder, muscle_name, with_noise, save_all = True) : 
    
    """Try hyperparameters, keep all train-evaluated models in a list and return best hyperparams
    
    INPUTS : 
    - Hyperparams : (ModelHyperparameters) all hyperparameters to try, choosen by user
    PLEASE, look at examples below
    - folder_name : string, path/name of the folder contained all excel data file of muscles (one for each muscle)
    
    OUTPUT : 
    - list_simulation : list of all hyperparameters try and results of train-eval (loss and acc)
    - best_hyperparameters : ModelHyperparameters, best hyperparameters (regarding of min val_loss)
    NOTE : best_hyperparameters is in the "single syntaxe", in that case, it is possible to use it with 
    "main_superised_learning" to save the model :)
    
    ---------
    Examples :
    ---------
    To avoid bug, please pay attention to syntax
    ONLY n_layers, n_nodes, activations, activation_names, L1_penalty, L2_penalty, learning_rate, and criterion could
    have multiple values
    
    n_layers = [1] or n_layers = [2] 
    n_nodes = [[12]] or n_nodes = [[12, 10], [12, 8]]
    activations = [[nn.GELU()]] or activations = [[nn.GELU(), nn.GELU()]]
    activation_names = [["GELU"]] or activation_names = [["GELU", "GELU"]]
    L_penalty = [0.01] or [0.01, 0.001]
    learning_rate = [1e-3] or [1e-3, 1e-4]

    criterion = [ModifiedHuberLoss(delta=[0.2], factor=[1.0])]
    or 
    criterion = [
        (LogCoshLoss, {'factor': [1.0, 1.5, 1.8]}),
        (ModifiedHuberLoss, {'delta': [0.2, 0.5, 1.0, 1.5, 2.0], 'factor': [1.0, 1.5, 2, 2.5, 3.0]}),
        (ExponentialLoss, {'alpha': [0.5, 0.8, 1.0]})
    ]
    p_dropout = [0.2] or [0.2, 0.5]   """
    
    # Before beggining, compute an estimation of execution time
    # The user can choose to stop if the execution is to long according to him 
    # For example, if estimed execution time if around 100 hours... maybe you have to many hyperparameters to try ...
    total_time_estimated_s, total_time_estimated_min, total_time_estimated_h = compute_time_testing_hyperparams(
        Hyperparams, time_per_configuration_secondes = 60)
    
    print(f"------------------------\n"
          f"Time estimated for testing all configurations: \n- {total_time_estimated_s} secondes"
          f"\n- {total_time_estimated_min} minutes\n- {total_time_estimated_h} hours\n\n"
          f"Research of best hyperparameters will beggining in few secondes ...\n"
          f"------------------------")
    time.sleep(0)
    
    print("Let's go !")
    # ------------------
    directory = f"{folder}/{muscle_name}/_Model/{Hyperparams.model_name}"
    create_directory(directory)

    folder_name = f"{folder}/{muscle_name}"
    train_loader, val_loader, test_loader, input_size, output_size, _ \
    = create_loaders_from_folder(Hyperparams, nbQ, num_datas_for_dataset, folder_name, muscle_name, 
                                 with_noise, plot = False)

    list_simulation= []
    best_val_loss = float('inf')
    best_criterion_class_loss = None
    best_criterion_params_loss = None
    num_try = 0
    
    for params in product(Hyperparams.n_layers, Hyperparams.n_nodes, Hyperparams.activations, Hyperparams.activation_names, 
                          Hyperparams.L1_penalty, Hyperparams.L2_penalty,Hyperparams.learning_rate, Hyperparams.dropout_prob):
        
        try_hyperparams = ModelHyperparameters("Try Hyperparams",Hyperparams.mode, Hyperparams.batch_size, params[0], 
                                               params[1], params[2], params[3], params[4], params[5], params[6], 
                                               Hyperparams.num_epochs, None, params[7], Hyperparams.use_batch_norm)
        
        for criterion_class, criterion_param_grid in Hyperparams.criterion:
            for criterion_params_comb in product(*criterion_param_grid.values()):
                criterion_params = dict(zip(criterion_param_grid.keys(), criterion_params_comb))
                try_hyperparams.add_criterion(criterion_class(**criterion_params))
                
                print(try_hyperparams)
                
                # Train-Evaluate model
                create_directory(f"{directory}/{num_try}")
                
                with measure_time() as timer:
                    # Please, consider this mesure time like as estimation 
                    _, val_loss, val_acc, epoch = train_model_supervised_learning(train_loader, val_loader, test_loader, 
                                                                            input_size, output_size, try_hyperparams, 
                                                                            file_path=f"{directory}/{num_try}", 
                                                                            plot = False, save = save_all)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_criterion_class_loss = criterion_class
                    best_criterion_params_loss = criterion_params
                    
                    # Best hyperparameters
                    best_hyperparameters_loss = ModelHyperparameters("Best_hyperparameter", Hyperparams.mode, 
                                                                     Hyperparams.batch_size, try_hyperparams.n_layers,
                                                                     try_hyperparams.n_nodes, try_hyperparams.activations, 
                                                                     try_hyperparams.activation_names, 
                                                                     try_hyperparams.L1_penalty, 
                                                                     try_hyperparams.L2_penalty, 
                                                                     try_hyperparams.learning_rate, Hyperparams.num_epochs, 
                                                                     try_hyperparams.criterion, try_hyperparams.dropout_prob, 
                                                                     Hyperparams.use_batch_norm)
                    best_hyperparameters_loss.save_results_parameters(val_loss, val_acc)

                list_simulation.append([val_loss, f"\nnum_try : {num_try} | val_loss = {val_loss} | val acc = {val_acc}",
                                        f"Time execution: {timer.execution_time:.6f} seconds ",
                                        f"Training with hyperparameters : {try_hyperparams} \n",
                                        f"Num of epoch used : {epoch + 1}",
                                        f"Criterion: {criterion_class.__name__}",
                                        f"with parameters: {criterion_params}\n----------\n"])
                num_try+=1

    list_simulation.sort(key=lambda x: x[0]) # sort list to have val_loss in croissant order 
    save_text_to_file('\n'.join([str(line) for sublist in list_simulation for line in sublist]), 
                      f"{folder}/{muscle_name}/_Model/{Hyperparams.model_name}/list_simulation.txt")
  
    print(f"Best hyperparameters loss found : {best_hyperparameters_loss}")
    print(f'Best criterion: {best_criterion_class_loss.__name__} with parameters: {best_criterion_params_loss}')
    print("list_simulation = ", list_simulation)
    
    # Finally, train one last time the model with best hyperparams and save model + plot
    main_superised_learning(best_hyperparameters_loss, nbQ, num_datas_for_dataset, folder, muscle_name, True,
                            f"{Hyperparams.model_name}/Best_hyperparams",with_noise, plot_preparation=True,plot=True,
                            save=True)
    
    return list_simulation, best_hyperparameters_loss


