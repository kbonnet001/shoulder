from neural_networks.data_preparation import get_x, get_y_and_labels
from neural_networks.save_model import main_function_model
import pandas as pd
import torch
from neural_networks.Mode import Mode
import copy
from neural_networks.muscle_forces_and_torque import compute_torque
import numpy as np
import matplotlib.pyplot as plt
from neural_networks.file_directory_operations import create_and_save_plot

def compare_model_torque_prediction(save_model_paths, modes, csv_path_datas) : 
    # Load data from CSV file
    df_datas = pd.read_csv(csv_path_datas) #.sort_values(by="torque")
    torque_predictions = []
    labels = []
    
    for save_model, mode in zip(save_model_paths, modes) : 
        # on recupere les entrees en se fiant au modele
        x = get_x(mode, df_datas, get_origin_and_insertion = False)
        # Convert to PyTorch tensors
        inputs = torch.tensor(x, dtype=torch.float32)
        _, y_labels = get_y_and_labels(mode, df_datas, get_y = False)
        outputs, _, _ = main_function_model(save_model, inputs)
        outputs = outputs.detach().numpy()
        labels.append(save_model.split("_Model/", 1)[1])
        
        if mode == Mode.TORQUE :
            # output deja egal a predicion torque, ob ajoute directement
            torque_predictions.append(outputs)
        elif mode == Mode.DLMT_DQ_F_TORQUE : 
            # torque quelque par dans les sorties
            torque_predictions.append(outputs[:, y_labels.index("torque")])
        elif mode == Mode.DLMT_DQ_FM : 
            # on a pas le torque, on doit le calculer manuellement
            # Récupérer les indices des colonnes dont le nom commence par "dlmt_dq_"
            index = [i for i, label in enumerate(y_labels) if label.startswith("dlmt_dq_")]

            # Extraire les colonnes correspondantes dans outputs
            dlmt_dq = outputs[:, index]
            fm = outputs[:, y_labels.index("muscle_force")]
            torque = [compute_torque(dlmt_dq_i, fm_i) for dlmt_dq_i, fm_i in zip(dlmt_dq, fm)]
            torque_predictions.append(np.array(torque))
            
        else : # mode doesn't exist
            raise ValueError(f"Invalid mode: {mode}. The mode does not exist or is not supported.")
        
    print(torque_predictions)
    
    # Plot the results
    plt.plot(df_datas.loc[:, "torque"].values[:100], marker='o', linestyle='-', label = "torque_target")
    for k in range (len(torque_predictions)) : 
        plt.plot(torque_predictions[k][:100], marker='o', linestyle='-', label = labels[k])
    plt.xlabel(f'n')
    plt.ylabel('Torque (Nm)')
    plt.title(f'Torque predictions - model comparaison')
    plt.grid(True)
    plt.legend()
    
    # Save and display the plot
    # create_and_save_plot(f"{csv_path_datas}", "model comparaison.png")
    plt.show()