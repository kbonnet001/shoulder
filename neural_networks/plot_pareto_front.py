import matplotlib as plt
import matplotlib.pyplot as plt
import numpy as np
import os
from neural_networks.file_directory_operations import create_and_save_plot, read_info_model
import copy
import pandas as pd
from neural_networks.CSVBatchWriterTestHyperparams import CSVBatchWriterTestHyperparams

def find_points_front_pareto(num_points, x_axis, y_axis) :
    pareto_indices = []
    for i in range(num_points):
        dominated = False
        for j in range(num_points):
            if (x_axis[j] <= x_axis[i] and y_axis[j] <= y_axis[i]) and (x_axis[j] < x_axis[i] or y_axis[j] < y_axis[i]):
                dominated = True
                break
        if not dominated:
            pareto_indices.append(i)
    return pareto_indices


def plot_results_try_hyperparams(csv_file, x_info, y_info, id = "num_try"):
    
    # Get informations for plot
    if os.path.exists(csv_file) : 
        df_datas = pd.read_csv(csv_file)
        x_axis = df_datas.loc[:, x_info].values
        y_axis = df_datas.loc[:, y_info].values
        model_id = df_datas.loc[:, id].values
    
    # Generate unique colors for each point using a colormap
    num_points = len(x_axis)
    colors = plt.cm.jet(np.linspace(0, 1, num_points))

    plt.figure(figsize=(10, 5))
    plt.xscale('log')
    plt.yscale('log')
    
    # Détection des points du front de Pareto
    pareto_indices = find_points_front_pareto(num_points, x_axis, y_axis)

    # Tracé des points
    for i in range(num_points):
        plt.scatter(y_axis[i], x_axis[i], marker='P', color=colors[i])
        if i in pareto_indices: 
            plt.scatter(y_axis[i], x_axis[i], edgecolor='black', facecolor='none', s=100)
            plt.text(y_axis[i], x_axis[i], model_id[i], fontsize=9, ha='right', weight='bold')
        else : 
            plt.text(y_axis[i], x_axis[i], model_id[i], fontsize=9, ha='right')

    # Tracer une ligne pour visualiser le front
    pareto_points = sorted([(x_axis[i], y_axis[i]) for i in pareto_indices])
    pareto_x, pareto_y = zip(*pareto_points)
    plt.plot(pareto_y, pareto_x, linestyle='--', color='black', alpha=0.6, label = "Pareto_front")
    plt.text(pareto_y[0] + 0.5, pareto_x[0] + 0.5, f"Best solution\n of objectif\n '{y_info}'", fontsize=9, ha='left', va='top', bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))
    plt.text(pareto_y[-1] - 0.5, pareto_x[-1] + 4, f"Best solution\n of objectif\n '{x_info}'", fontsize=9, ha='right', va='bottom', bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))
    
    plt.xlabel(f"{x_info}" + (" (s)" if x_info.startswith("execution_time")else ""))
    plt.ylabel(f"{y_info}" + (" (s)" if y_info.startswith("execution_time") else ""))
    plt.title(f"{x_info} vs {y_info}", fontweight='bold')
    plt.grid(True)
    plt.legend()
    create_and_save_plot(f"{os.path.dirname(csv_file)}", f"{x_info} vs {y_info}.png")
    plt.show()

def plot_results_try_hyperparams_comparaison(dir_paths, x_info, y_info, save_path, id = "num_try"):
    # dir_paths = [directory_path_1, directory_path_2]
    x_axis = []
    y_axis = []
    model_id = []
    pareto_indices = []
    
    # Get informations for plot
    for full_directory_path in dir_paths:
        # full_directory_path = os.path.join(dir_paths, csv_file)
        if os.path.exists(f"{full_directory_path}") :
            df_datas = pd.read_csv(full_directory_path)
            x_axis.append(df_datas.loc[:, y_info].values) 
            y_axis.append(df_datas.loc[:, x_info].values)
            model_id.append(df_datas.loc[:, id].values)
    
    # Generate unique colors for each point using a colormap
    num_colors = len(dir_paths)
    colors = plt.cm.jet(np.linspace(0, 1, num_colors))
    markers = ["P", "^", "o"]

    plt.figure(figsize=(10, 5))
    plt.xscale('log')
    plt.yscale('log')
    file_names = [path.split('/')[-1] for path in dir_paths]
    # num_points = list(chain.from_iterable(x_axis))
    
    # Détection des points du front de Pareto
    # for n in range (len(dir_paths)) : 
    #     pareto_indices.append(find_points_front_pareto(len(x_axis[n]), x_axis[n], y_axis[n]))


    # for x, y, m_id in zip(x_axis, y_axis, model_id) : 
    #     for i in range(len(x_axis[0])):
    #         if i == 0 :
    #             plt.scatter(y[i], x[i], marker='P', color=colors[i], label = file_names[i])
    #         else : 
    #             plt.scatter(y[i], x[i], marker='P', color=colors[i])
    #         plt.text(y[i], x[i], m_id[i], fontsize=9, ha='right')

    # # Tracé des points
    for i in range(len(dir_paths)) : 
        for j in range(len(x_axis[i])): #len(x_axis[i])
            if j == 0 : 
                plt.scatter(y_axis[i][j], x_axis[i][j], marker=markers[i], color=colors[i], label = file_names[i])
            else : 
                plt.scatter(y_axis[i][j], x_axis[i][j], marker=markers[i], color=colors[i])
            plt.text(y_axis[i][j], x_axis[i][j], model_id[i][j], fontsize=9, ha='right')

            # if j in pareto_indices[i]: 
            #     plt.scatter(y_axis[i][j], x_axis[i][j], edgecolor='black', facecolor='none', s=100)
            #     plt.text(y_axis[i][j], x_axis[i][j], model_names_try[i][j], fontsize=9, ha='right', weight='bold')
            # else : 
            

    # Tracer une ligne pour visualiser le front
    # pareto_points = sorted([(x_axis[i], y_axis[i]) for i in pareto_indices])
    # pareto_x, pareto_y = zip(*pareto_points)
    # plt.plot(pareto_y, pareto_x, linestyle='--', color='black', alpha=0.6, label = "Pareto_front")
    # plt.text(pareto_y[0] + 0.5, pareto_x[0] + 0.5, f"Best solution\n of objectif\n '{y_info}'", fontsize=9, ha='left', va='top', bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))
    # plt.text(pareto_y[-1] - 0.5, pareto_x[-1] + 4, f"Best solution\n of objectif\n '{x_info}'", fontsize=9, ha='right', va='bottom', bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))
    
    plt.xlabel(f"{x_info}" + (" (s)" if x_info == "execution_time" else ""))
    plt.ylabel(f"{y_info}" + (" (s)" if y_info == "execution_time" else ""))
    plt.title(f"{x_info} vs {y_info}", fontweight='bold')
    plt.grid(True)
    plt.legend()
    text_file_names = " vs ".join([file_name.split('.')[0] for file_name in file_names])
    create_and_save_plot(f"{save_path}", f"{text_file_names}.png")
    plt.show()

def create_df_from_txt_saved_informations(directory_csv) : 
    writer = CSVBatchWriterTestHyperparams(f"{directory_csv}", batch_size=100)
    
    for directory in os.listdir(f"{os.path.dirname(directory_csv)}"):
        full_directory_path = os.path.join(f"{os.path.dirname(directory_csv)}", directory)
        if os.path.isdir(full_directory_path) and os.path.exists(f"{full_directory_path}/model_informations.txt") :
            # x_axis_value, y_axis_value = read_info_model(f"{full_directory_path}/model_informations.txt", [x_info, y_info])
            num_try, val_loss, val_acc, train_timer, mean_model_load_timer, \
            mean_model_timer, mode, batch_size, n_nodes, activations, L1_penalty, L2_penalty, \
            learning_rate, dropout_prob, use_batch_norm, epoch, criterion_name, criterion_params \
            = read_info_model(f"{full_directory_path}/model_informations.txt", 
                              ["num_try", "val_loss", "val_acc", "execution_time_train", "execution_time_load_saved_model", 
                               "execution_time_use_saved_model", "mode", "batch_size", "n_nodes", "activations", "L1_penalty", 
                               "L2_penalty", "learning_rate", "dropout_prob", "use_batch_norm", "num_epochs_used", "criterion_name", 
                               "criterion_params"])
            
            writer.add_line_full(num_try, val_loss, val_acc, train_timer, mean_model_load_timer, 
                 mean_model_timer, batch_size, n_nodes, activations, L1_penalty, L2_penalty, learning_rate, 
                 dropout_prob, use_batch_norm, mode, epoch, criterion_name, criterion_params)
    
    writer.close()
            
            