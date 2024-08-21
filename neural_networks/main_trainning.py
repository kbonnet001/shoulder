from neural_networks.data_preparation import create_loaders_from_folder, dataloader_to_tensor
from neural_networks.data_tranning import train_model_supervised_learning
from neural_networks.activation_functions import *
import os
from neural_networks.save_model import measure_time, save_model, main_function_model, del_saved_model
from neural_networks.plot_visualisation import visualize_prediction_training, visualize_prediction, mean_distance, compute_pourcentage_error
from itertools import product
from neural_networks.Loss import *
from neural_networks.ModelHyperparameters import ModelHyperparameters
from neural_networks.file_directory_operations import create_directory, save_text_to_file,save_informations_model, get_min_value_from_csv
import time
from neural_networks.plot_pareto_front import plot_results_try_hyperparams
import numpy as np
from neural_networks.CSVBatchWriterTestHyperparams import CSVBatchWriterTestHyperparams

def compute_time_testing_hyperparams(Hyperparams, time_per_configuration_secondes=60):
    """Compute an estimation of execution time for testing hyperparameter configurations.
    This estimation is linear and not very accurate, serving as a rough guideline.
    
    Args:
    - Hyperparams: ModelHyperparameters, containing all hyperparameters to try as specified by the user.
    - time_per_configuration_secondes: float, (default 60), estimated time in seconds to train and evaluate one model configuration.
    
    Returns:
    - total_time_estimated_secondes: float, total estimated execution time in seconds.
    - total_time_estimated_minutes: float, total estimated execution time in minutes.
    - total_time_estimated_hours: float, total estimated execution time in hours.
    """

    # Calculate the number of hyperparameter combinations to test.
    n_combinations = (
        len(Hyperparams.n_nodes) *
        len(Hyperparams.activations) *
        len(Hyperparams.L1_penalty) *
        len(Hyperparams.L2_penalty) *
        len(Hyperparams.learning_rate) *
        len(Hyperparams.dropout_prob) *
        sum(len(list(product(*params.values()))) for _, params in Hyperparams.criterion)
    )

    # Estimate the total time for all combinations in seconds
    total_time_estimated_secondes = n_combinations * time_per_configuration_secondes
    
    # Convert the total time from seconds to minutes and hours
    total_time_estimated_minutes = total_time_estimated_secondes / 60
    total_time_estimated_hours = total_time_estimated_secondes / 3600
    
    return total_time_estimated_secondes, total_time_estimated_minutes, total_time_estimated_hours

def update_best_hyperparams(model, num_try, hyperparams_i, try_hyperparams_ref, input_size, output_size, directory) : 
    """
    Update the best hyperparameters and save the corresponding model.

    Args:
    - model: The trained model that performed best with the current hyperparameters.
    - num_try : Num id of the model 
    - hyperparams_i: The current instance of ModelHyperparameters that yielded the best performance.
    - try_hyperparams_ref: Reference instance of ModelTryHyperparameters containing common settings.
    - input_size: The size of the input layer of the model.
    - output_size: The size of the output layer of the model.
    - directory: Directory path where the best model will be saved.

    Returns:
    - best_hyperparameters_loss: Instance of ModelHyperparameters with the best hyperparameter settings.
    """
    # Create a new ModelHyperparameters instance to store the best hyperparameter settings.
    best_hyperparameters_loss = ModelHyperparameters(num_try, try_hyperparams_ref.batch_size,
                                                        hyperparams_i.n_nodes, hyperparams_i.activations, 
                                                        hyperparams_i.activation_names, 
                                                        hyperparams_i.L1_penalty, 
                                                        hyperparams_i.L2_penalty, 
                                                        hyperparams_i.learning_rate, try_hyperparams_ref.num_epochs, 
                                                        hyperparams_i.criterion, hyperparams_i.dropout_prob, 
                                                        try_hyperparams_ref.use_batch_norm)

    # Save the best model
    save_model(model, input_size, output_size, best_hyperparameters_loss, 
                f"{directory}/Best_hyperparams") 
    return best_hyperparameters_loss

def compute_mean_model_timers(file_path, all_data_tensor) : 
    """ Compute the mean execution times for loading and running models.
    
    Args:
    - file_path: str, path to the file containing model data or configuration.
    - all_data_tensor: list, a collection of data tensors to be processed by the models.
    
    Returns:
    - mean_model_load_timer: float, average time taken to load the model across all data tensors.
    - mean_model_timer: float, average time taken to execute the model across all data tensors.
    """
    model_load_timers = []
    model_timers = []
    for n in range(len(all_data_tensor)) : 
        # Call the main function to process the model with the current data tensor
        # and retrieve the loading and execution timers
        _, model_load_timer, model_timer = main_function_model(file_path, all_data_tensor[n]) 
        # Append the execution times to the corresponding lists
        model_load_timers.append(model_load_timer.execution_time)
        model_timers.append(model_timer.execution_time)
        
    # Calculate the mean execution time 
    mean_model_load_timer = np.mean(model_load_timers)
    mean_model_timer = np.mean(model_timers)
    
    return mean_model_load_timer, mean_model_timer

def get_num_try(directory_path):
    """
    Function to determine the next numeric directory name to use.
    
    This function scans the specified directory for subdirectories with numeric names
    (e.g., '0', '1', '2', etc.), determines the highest number, and returns the next 
    number in the sequence (i.e., the maximum number found plus one).

    Args:
        directory_path (str): The path to the directory containing the numbered subdirectories.

    Returns:
        int: The next number to be used as a directory name.
    
    Raises:
        AssertionError: If no numbered directories are found in the specified path.
    """
    
    # Get the list of directories in the specified path
    dirs = [d for d in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, d))]

    # Filter to keep only directories whose names are numeric
    numbered_dirs = [int(d) for d in dirs if d.isdigit()]

    if numbered_dirs:
        # Find the maximum number among the numeric directories
        max_n = max(numbered_dirs)
    else:
        # If no numbered directory is found, raise an error
        raise AssertionError(f"No numbered directories found in the specified path.",
                             f"Ckeck you directory : {directory_path}.",
                             f"Try to delete the directory and try again.")

    # Return the next number in the sequence (max_n + 1)
    return max_n + 1

def main_supervised_learning(Hyperparams, mode, nb_q, nb_segment, nb_muscle, num_datas_for_dataset, folder_name, muscle_name, retrain, 
                            file_path, with_noise, plot_preparation, plot_loss_acc, plot_loader, save) : 

    """ Main function to prepare, train, validate, test, and save a model.
    
    Args:
    - Hyperparams: ModelHyperparameters, all hyperparameters chosen by the user.
        To avoid bugs, please pay attention to syntax. More details in ModeHyperparameters.py
    - mode: Mode for the operation
    - nb_q: int, number of q (generalized coordinates) in the biorbd model.
    - nb_segment: int, number of segment in the biorbd model.
    - nb_muscle: int, number of muscle in the biorbd model.
    - num_datas_for_dataset: int, number of data points for the dataset used for training.
    - folder_name: str, path/name of the folder containing all CSV data files for muscles (one for each muscle).
    - muscle_name: str, name of the muscle.
    - retrain: bool, True to train the model again.
    - file_path: str, the path where the model will be saved after training.
    - with_noise: bool, (default = True), True to include noisy data in the dataset for learning.
    - plot_preparation: bool, True to show the distribution of all data preparation.
    - plot_loss_acc: bool, True to show plot loss, accuracy, predictions/targets.
    - plot_loader: bool, True to show plot comparing the loader data (train, validation, test) with the target data
        Warning : This plot is not available if the output height is set too large, as there may be too many subplots to display!
    - save: bool, True to save the model to file_path.
    """
    
    # Create a folder for save plots
    folder_name_muscle = f"{folder_name}/{muscle_name}"
    create_directory(f"{folder_name_muscle}/_Model") # Muscle/Model
    
    # Train_model if retrain == True or if none file_path already exist
    if retrain or os.path.exists(f"{folder_name_muscle}/_Model/{file_path}")==False : 
        create_directory(f"{folder_name_muscle}/_Model/{file_path}") # Muscle/Model
        
        # Prepare datas for trainning
        train_loader, val_loader, test_loader, input_size, output_size, y_labels \
         = create_loaders_from_folder(Hyperparams, mode, nb_q, nb_segment, num_datas_for_dataset, f"{folder_name_muscle}", 
                                      muscle_name, with_noise, plot_preparation)
        # Trainning
        model, _, _, _, _, _ = train_model_supervised_learning(train_loader, val_loader, test_loader, input_size, 
                                                               output_size, Hyperparams, 
                                                               f"{folder_name_muscle}/_Model/{file_path}", plot_loss_acc, save, 
                                                               show_plot=True)
        if plot_loader : 
            # Visualize tranning : predictions/targets for loaders train, val and test
            visualize_prediction_training(model, f"{folder_name_muscle}/_Model/{file_path}", y_labels, train_loader,
                                        val_loader, test_loader) 
    # Visualize : predictions/targets for all q variation
    visualize_prediction(mode, Hyperparams.batch_size, nb_q, nb_segment, nb_muscle, f"{folder_name_muscle}/_Model/{file_path}", 
                         f"{folder_name_muscle}/plot_all_q_variation_")
    
def find_best_hyperparameters(try_hyperparams_ref, mode, nb_q, nb_segment, nb_muscle, num_datas_for_dataset, folder, muscle_name, 
                              with_noise, save_all = False) : 
    
    """Try hyperparameters, keep all train-evaluated models in a list, and return the best hyperparameters.
    
    Args:
    - try_hyperparams_ref: ModelTryHyperparameters, all hyperparameters to try, chosen by the user.
    - mode: mode for the operation, could be a string or an identifier related to the data processing or model setup.
    - nb_q: int, number of q (generalized coordinates) in the biorbd model.
    - nb_segment: int, number of segment in the biorbd model.
    - nb_muscle: int, number of muscle in the biorbd model.
    - num_datas_for_dataset: int, number of data points for the dataset used for training.
    - folder: str, path/name of the folder containing all CSV data files for muscles (one for each muscle).
    - muscle_name: str, name of the muscle.
    - with_noise: bool, True to train with data that includes noise, False to train with only pure data.
    - save_all: bool, (default = False) True to save all tested models. 
      Be cautious as saving all models can be heavy, especially if n_nodes are large. 
      The best model (in terms of validation loss) will always be saved.

    Return:
    - best_hyperparameters: ModelHyperparameters, best hyperparameters (in terms of minimum validation loss).
      NOTE: best_hyperparameters is in the "single syntax". In this case, it is possible to use it with 
      "main_supervised_learning" with retrain = False for example.
    """

    # Before beggining, compute an estimation of execution time
    # The user can choose to stop if the execution is to long according to him 
    # For example, if estimed execution time if around 100 hours... maybe you have to many hyperparameters to try ...
    total_time_estimated_s, total_time_estimated_min, total_time_estimated_h = compute_time_testing_hyperparams(
        try_hyperparams_ref, time_per_configuration_secondes = 60)
    
    print(f"------------------------\n"
          f"Time estimated for testing all configurations: \n- {total_time_estimated_s} seconds"
          f"\n- {total_time_estimated_min} minutes\n- {total_time_estimated_h} hours\n\n"
          f"Research of best hyperparameters will begin in 10 seconds...\n"
          f"------------------------")
    time.sleep(0)
    
    print("Let's go !")
    # ------------------
    
    directory = f"{folder}/{muscle_name}/_Model/{try_hyperparams_ref.model_name}"
    if os.path.exists(f"{f"{directory}/Best_hyperparams"}") and os.listdir(f"{f"{directory}/Best_hyperparams"}"): 
        # Get num_try
        num_try = get_num_try(directory)
        best_val_loss = get_min_value_from_csv(f"{directory}/{try_hyperparams_ref.model_name}.CSV", "val_loss")
    else : 
        # Create directory to save all test
        create_directory(f"{directory}/Best_hyperparams")
        num_try = 0
        best_val_loss = float('inf')

    # Create loaders for trainning
    folder_name = f"{folder}/{muscle_name}"
    train_loader, val_loader, test_loader, input_size, output_size, _ \
    = create_loaders_from_folder(try_hyperparams_ref, mode, nb_q, nb_segment, num_datas_for_dataset, folder_name, muscle_name, 
                                 with_noise, plot = False)
    
    all_data_test_tensor, _ = dataloader_to_tensor(test_loader)
    
    writer = CSVBatchWriterTestHyperparams(f"{directory}/{try_hyperparams_ref.model_name}.CSV", batch_size=100)

    best_criterion_class_loss = None
    best_criterion_params_loss = None
    
    # Loop to try all configurations of hyperparameters
    for params in product(try_hyperparams_ref.n_nodes, try_hyperparams_ref.activations, 
                          try_hyperparams_ref.activation_names, try_hyperparams_ref.L1_penalty, 
                          try_hyperparams_ref.L2_penalty,try_hyperparams_ref.learning_rate, 
                          try_hyperparams_ref.dropout_prob):
        
        hyperparams_i = ModelHyperparameters("Try Hyperparams", try_hyperparams_ref.batch_size, 
                                               params[0], params[1], params[2], params[3], params[4], params[5], 
                                               try_hyperparams_ref.num_epochs, None, params[6], 
                                               try_hyperparams_ref.use_batch_norm)
        
        for criterion_class, criterion_param_grid in try_hyperparams_ref.criterion:
            for criterion_params_comb in product(*criterion_param_grid.values()):
                criterion_params = dict(zip(criterion_param_grid.keys(), criterion_params_comb))
                hyperparams_i.add_criterion(criterion_class(**criterion_params))
                
                print(hyperparams_i)
                
                # Train-Evaluate model
                create_directory(f"{directory}/{num_try}")
                
                with measure_time() as train_timer: # timer --> trainning time
                    # Please, consider this mesure time as an estimation !
                    model, val_loss, test_acc, test_error, test_abs_error, epoch \
                    = train_model_supervised_learning(train_loader, val_loader, test_loader, 
                                                      input_size, output_size, hyperparams_i, 
                                                      file_path=f"{directory}/{num_try}", plot = True, save = True, 
                                                      show_plot=False) # save temporaly 
                # Timer for load model and model use
                # Mean with data_test_tensor (20% of num_datas_for_dataset)
                mean_model_load_timer, mean_model_timer \
                    = compute_mean_model_timers(f"{directory}/{num_try}", all_data_test_tensor)
                
                if save_all == False : 
                    # deleted saved model
                    del_saved_model(f"{directory}/{num_try}")
                
                # Check if these hyperparameters are the best
                if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_criterion_class_loss = criterion_class
                        best_criterion_params_loss = criterion_params
                        # Update the best hyperparameters
                        best_hyperparameters_loss = update_best_hyperparams(model, num_try, hyperparams_i, try_hyperparams_ref, 
                                                                            input_size, output_size, directory)
                
                save_informations_model(f"{directory}/{num_try}", num_try, val_loss, test_acc, test_error, test_abs_error,
                                        train_timer.execution_time, mean_model_load_timer, mean_model_timer,
                                        hyperparams_i, mode, epoch+1, criterion_class.__name__, criterion_params)
                
                writer.add_line(num_try, val_loss, test_acc, test_error, test_abs_error, train_timer.execution_time, 
                                mean_model_load_timer, mean_model_timer, hyperparams_i, mode, epoch+1, 
                                criterion_class.__name__, criterion_params)
                
                num_try+=1
                
    writer.close()
  
    print(f"Best hyperparameters loss found : {best_hyperparameters_loss}")
    print(f'Best criterion: {best_criterion_class_loss.__name__} with parameters: {best_criterion_params_loss}')
    
    # Plot visualisation to compare all model trained (pareto front)
    plot_results_try_hyperparams(f"{directory}/{try_hyperparams_ref.model_name}.CSV", "execution_time_train", "val_loss")
    plot_results_try_hyperparams(f"{directory}/{try_hyperparams_ref.model_name}.CSV", "execution_time_load_saved_model", 
                                 "val_loss")
    plot_results_try_hyperparams(f"{directory}/{try_hyperparams_ref.model_name}.CSV", "execution_time_use_saved_model", 
                                 "val_loss")
    
    # Finally, plot figure predictions targets with the best model saved

    main_supervised_learning(best_hyperparameters_loss, mode, nb_q, nb_segment, nb_muscle, num_datas_for_dataset, folder, muscle_name, False,
                            f"{try_hyperparams_ref.model_name}/Best_hyperparams",with_noise, plot_preparation=True, 
                            plot_loss_acc=True, plot_loader=False, save=True)
    
    return best_hyperparameters_loss


