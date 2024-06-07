import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import norm
from wrapping.plot_cylinder import *
from wrapping.algorithm import*
from wrapping.step_1 import find_cylinder_frame, find_matrix
from wrapping.Cylinder import Cylinder

from sklearn.model_selection import train_test_split
from neural_networks.main_trainning import test_model_supervised_learning

from neural_networks.data_generation import data_for_learning, test_limit_data_for_learning, data_for_learning_plot

#################### 
# Code des tests
import biorbd
import bioviz

model = biorbd.Model("models/Wu_DeGroote.bioMod")
q = np.zeros((model.nbQ(), ))

# Noms de tous les segments (= les os) du modèle
segment_names = [model.segment(i).name().to_string() for i in range(model.nbSegment())]
humerus_index = segment_names.index("humerus_right") # pour trouver où est humerus_index --» 6

humerus_dof_names = [model.segment(humerus_index).nameDof(i).to_string() for i in range(model.segment(humerus_index).nbQ())]
q_ranges = [[ranges.min(), ranges.max()] for ranges in model.segment(humerus_index).QRanges()]
q_ranges.append([0.05, 2.3561])

# segment_names
# ['thorax', 'spine', 'clavicle_effector_right', 'clavicle_right', 'scapula_effector_right', 'scapula_right', 'humerus_right', 'ulna_effector_right', 'ulna_right', 'radius_effector_right', 'radius_right', 'hand_right']

# INPUTS :  
# --------

# Datas pour le cylindre (à priori) du thorax pour PECM2 et PECM3 (à partir de deux points)
C_T_PECM2_1 = np.array([0.0187952455, -0.0992656744, 0.0784311931])
C_T_PECM2_2 = np.array([0.0171630409, -0.014154527, 0.0749634019])

# Datas for cylinder's humerus right (muscle PECM1, PECM2 and PCM3)

C_H_PECM2_1 = np.array([-0.0468137093, -0.069205313, 0.1748923225])
C_H_PECM2_2 = np.array([-0.0276992818, 0.0056711748, 0.1704452973])

cylinder_T_PECM2 = Cylinder.from_points(0.025, -1, C_T_PECM2_1, C_T_PECM2_2, "thorax")
cylinder_H_PECM2 = Cylinder.from_points(0.02, 1, C_H_PECM2_1, C_H_PECM2_2, "humerus_right")

cylinders_PECM2=[cylinder_T_PECM2, cylinder_H_PECM2]

muscle_selected = "PECM2"

# test_limit_data_for_learning(muscle_selected,cylindekrs_PECM2, model, q_ranges,"PECM2_datas00.xlsx", True) 

# data_for_learning (muscle_selected,cylinders_PECM2, model, q_ranges, 1, "df_PECM2_datas_5000.xlsx") 
# ----------------------
test_model_supervised_learning("df_PECM2_datas_5000.xlsx")
# ----------------------
# data_for_learning_plot (muscle_selected, cylinders_PECM2, model, q_ranges, 2, 100, plot_all=False, plot_limit=False)


# # Show
# b = bioviz.Viz(loaded_model=model)
# b.set_q(q)
# b.exec()

# exit(0)

#################