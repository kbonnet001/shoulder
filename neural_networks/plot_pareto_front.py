import matplotlib as plt
import matplotlib.pyplot as plt
import numpy as np
import os
from neural_networks.file_directory_operations import create_and_save_plot, read_info_model
import copy

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


def plot_results_try_hyperparams(directory_path, x_info, y_info):
    x_axis = []
    y_axis = []
    model_name_try = []
    
    # Get informations for plot
    for directory in os.listdir(directory_path):
        full_directory_path = os.path.join(directory_path, directory)
        if os.path.isdir(full_directory_path) and os.path.exists(f"{full_directory_path}/model_informations.txt") :
            x_axis_value, y_axis_value = read_info_model(f"{full_directory_path}/model_informations.txt", [x_info, y_info])
            x_axis.append(x_axis_value)
            y_axis.append(y_axis_value)
            model_name_try.append(directory) 
    
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
            plt.text(y_axis[i], x_axis[i], model_name_try[i], fontsize=9, ha='right', weight='bold')
        else : 
            plt.text(y_axis[i], x_axis[i], model_name_try[i], fontsize=9, ha='right')

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
    create_and_save_plot(f"{directory_path}", f"{x_info} vs {y_info}.png")
    plt.show()

def plot_results_try_hyperparams_comparaison(dir_paths, x_info, y_info, save_path):
    # dir_paths = [directory_path_1, directory_path_2]
    x_axis = []
    y_axis = []
    model_names_try = []
    pareto_indices = []
    
    # Get informations for plot
    for k in range(len(dir_paths)) : 
        x_ax = []
        y_ax = []
        model_name_try = []
        for directory in os.listdir(dir_paths[k]):
            full_directory_path = os.path.join(dir_paths[k], directory)
            if os.path.isdir(full_directory_path) and os.path.exists(f"{full_directory_path}/model_informations.txt") :
                x_axis_value, y_axis_value = read_info_model(f"{full_directory_path}/model_informations.txt", [x_info, y_info])
                x_ax.append(x_axis_value)
                y_ax.append(y_axis_value)
                model_name_try.append(directory) 
        x_axis.append(copy.deepcopy(x_ax))
        y_axis.append(copy.deepcopy(y_ax))
        model_names_try.append(copy.deepcopy(model_name_try))

    
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

    # Tracé des points
    for i in range(len(dir_paths)) : 
        for j in range(len(x_axis[i])):
            if j == 0 : 
                plt.scatter(y_axis[i][j], x_axis[i][j], marker=markers[i], color=colors[i], label = file_names[i])
            else : 
                plt.scatter(y_axis[i][j], x_axis[i][j], marker=markers[i], color=colors[i])
            plt.text(y_axis[i][j], x_axis[i][j], model_names_try[i][j], fontsize=9, ha='right')

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
    create_and_save_plot(f"{save_path}", f"{file_names[0]} vs {file_names[1]}.png")
    plt.show()
