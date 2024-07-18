import numpy as np
from scipy.linalg import norm
from wrapping.plot_cylinder import *
from wrapping.algorithm import*
from wrapping.Cylinder import Cylinder
from neural_networks.discontinuities import *
import torch.nn as nn
from neural_networks.Loss import *
from pyorerun import LiveModelAnimation

from neural_networks.data_generation import *
from neural_networks.ModelHyperparameters import ModelHyperparameters
from neural_networks.data_generation_ddl import data_for_learning_ddl, plot_one_q_variation, plot_all_q_variation, data_for_learning_without_discontinuites_ddl, data_generation_muscles
from neural_networks.k_cross_validation import cross_validation, try_best_hyperparams_cross_validation
from neural_networks.functions_data_generation import compute_q_ranges
from wrapping.lever_arm import plot_lever_arm
from neural_networks.Mode import Mode
from neural_networks.main_trainning import main_superised_learning, find_best_hyperparameters

#################### 
# Code des tests
import biorbd
import bioviz

import unittest

# Importer les tests
from wrapping.wrapping_tests.step_1_test import Step_1_test
# from wrapping.wrapping_tests.step_2_test import Step_2_test

# unittest.main()
###############################################
###############################################

model_biorbd = biorbd.Model("models/Wu_DeGroote.bioMod")

q_ranges, _ = compute_q_ranges(model_biorbd)


# INPUTS :  
# --------

# Datas pour le cylindre (à priori) du thorax pour PECM2 et PECM3 (à partir de deux points)
C_T_PECM2_1 = np.array([0.0183539873, -0.0762563082, 0.0774936934])
C_T_PECM2_2 = np.array([0.0171218365, -0.0120059285, 0.0748758588])

C_H_PECM2_1 = np.array([-0.0504468139, -0.0612220954, 0.1875298764])
C_H_PECM2_2 = np.array([-0.0367284615, -0.0074835226, 0.1843382632]) #le mieux avec 0.025 0.0243 vrai 0.0255913399
# -----------------------------------------------------------------
cylinder_T_PECM2 = Cylinder.from_points(0.025, -1, C_T_PECM2_2, C_T_PECM2_1, False, "thorax", "PECM2")
cylinder_H_PECM2 = Cylinder.from_points(0.0255913399, 1, C_H_PECM2_2, C_H_PECM2_1, True, "humerus_right", "PECM2")

C_T_PECM3_1 = np.array([0.0191190885, -0.1161524375, 0.0791192319])
C_T_PECM3_2 = np.array([0.0182587352, -0.0712893992, 0.0772913203])

C_H_PECM3_1 = np.array([-0.0504468139, -0.0612220954, 0.1875298764])
C_H_PECM3_2 = np.array([-0.0367284615, -0.0074835226, 0.1843382632])

cylinder_T_PECM3 = Cylinder.from_points(0.025, -1, C_T_PECM3_2, C_T_PECM3_1, False, "thorax","PECM3")
cylinder_H_PECM3 = Cylinder.from_points(0.0202946443, 1, C_H_PECM3_2, C_H_PECM3_1, True, "humerus_right", "PECM3")

# cylinder_H_PECM2.rotate_around_axis(-45)
# -----------------------------------------------------------------
cylinders_PECM2=[cylinder_T_PECM2, cylinder_H_PECM2]
cylinders_PECM3=[cylinder_T_PECM3, cylinder_H_PECM3]

cylinders = [cylinders_PECM2, cylinders_PECM3]

muscles_selected = ["PECM2", "PECM3"]
segments_selected = ["thorax", "humerus_right"] # pour le moment, on change rien
# -----------------------------------------------------------------

# test_limit_data_for_learning(muscles_selected[0],cylinders_PECM2, model_biorbd, q_ranges, True, False) 

# data_for_learning (muscles_selected[0],cylinders_PECM2, model_biorbd, q_ranges, 5000, "df_PECM2_datas_without_error_partfdsadaf_5000.xlsx", False, False) 

# data_for_learning_ddl (muscles_selected[0], cylinders_PECM2, model_biorbd, 10, "rgtrsfdd.xlsx", data_without_error = True, plot=False, plot_cadran = False)

# -----------------------------------------------------------------

# q_fixed = np.array([(ranges[0] + ranges[-1]) / 2  for ranges in q_ranges])
q_fixed = np.array([0.0 for k in range (10)])
# q_fixed = np.array([(ranges[1]) for ranges in q_ranges])
# q_fixed = np.array([q_ranges[0][0], q_ranges[1][1], q_ranges[2][0], 0.0])
# q_fixed = np.array([(ranges[0] + ranges[-1]) / 2  for ranges in q_ranges_PECM2])

# plot_one_q_variation(muscles_selected[1], cylinders_PECM3, model_biorbd, q_fixed, 
#                         1, "PECM3_q1", 50, plot_all=False, plot_limit=True, plot_cadran=False)

# plot_all_q_variation(muscles_selected[0], cylinders_PECM2, model_biorbd, q_fixed, "PECM2_q_initial", num_points = 100, 
#                      plot_all = False, plot_limit = False, plot_cadran=False, file_path="data_generation_data_more_ddl_6/PECM2")

# Generate datas : 
#----------------
# data_for_learning_without_discontinuites_ddl(muscles_selected[0], cylinders[0], model_biorbd, 5010, "data_generation_data_more_ddl_6/PECM2", num_points = 100, plot_cylinder_3D=False, plot_discontinuities = False, plot_cadran = False, plot_graph=True)

# data_generation_muscles(muscles_selected, cylinders, model_biorbd, 5000, "datas_with_dlmt_dq", num_points = 20, plot_cylinder_3D=False, plot_discontinuities = False, plot_cadran = False, plot_graph=False)


# --------------------
# data_for_learning_without_discontinuites(muscles_selected, cylinders, model_biorbd, q_ranges, 5000, 
#                 "df_PECM3_datas_without_error_part_5000.xlsx", num_points = 50, plot_discontinuities = False, 
#                 plot=False, plot_cadran = False)
   
# -----------------------------------------------------------------

P = np.array([-3.78564,-2.53658,0])
S = np.array([7.0297,1.44896,1.21311])

c11 = np.array([0,-1,-4])
c12 = np.array([0,-1,4])

c21 = np.array([5.45601,-2.71188,-1.38174])
c22 = np.array([2.23726,4.56496,4])

cylinder_1 = Cylinder.from_points(1,-1, c11, c12)
cylinder_2 = Cylinder.from_points(1,-1, c21, c22)

# double_cylinder_obstacle_set_algorithm(P, S, cylinder_1, cylinder_2, np.dot(np.linalg.inv(cylinder_1.matrix), 
#                                                                             cylinder_2.matrix) )

# v1o, v2o, obstacle_tangent_point_inactive, segment_length = single_cylinder_obstacle_set_algorithm(P, S, cylinder_1)
# plot_one_cylinder_obstacle(P, S, cylinder_1, v1o, v2o,obstacle_tangent_point_inactive)

# -----------------------------------------------------------------
# Show biorbd
# q = np.zeros((model_biorbd.nbQ(), ))
# b = bioviz.Viz(loaded_model=model_biorbd)
# b.set_q(q)
# b.exec()

# exit(0)

# # pour voir pyorerun
# model_path = "/home/lim/Documents/kloe/shoulder/run/models/Wu_DeGroote.bioMod"
# animation = LiveModelAnimation(model_path, with_q_charts=True)
# animation.rerun()

# -----------------------------------------------------------------

# data_loaders = prepare_data_from_folder(32, "datas", plot=False)
# print("")

# model_name = "train_muscle_PECM2"
# mode = Mode.MUSCLE
# batch_size = 32
# n_layers = [1]
# n_nodes = [[8], [10], [12], [15], [20], [25], [30]]
# activations = [[nn.GELU()]]
# activation_names = [["GELU"]]
# L1_penalty = [0.01, 0.001]
# L2_penalty = [0.01, 0.001]
# learning_rate = [1e-3]
# num_epochs = 1000
# # criterion = ModifiedHuberLoss(delta=0.2, factor=1.0)
# criterion= [
#     (LogCoshLoss, {'factor': [1.0, 1.8]}),
#     (ModifiedHuberLoss, {'delta': [0.2, 1.0, 2.0], 'factor': [1.0, 2.0, 3.0]}),
#     (ExponentialLoss, {'alpha': [0.5, 1.0]})
# ]
# p_dropout = [0.2, 0.5]
# use_batch_norm = True

# model_name="essai_dlmt_dq_lmt_x"
# mode = Mode.DLMT_DQ
# batch_size=128
# n_layers=1
# n_nodes=[128, 64, 32]
# activations=[nn.GELU(), nn.GELU(), nn.GELU()]
# activation_names = ["GELU", "GELU", "GELU"]

model_name="essai_muscle0" #0 meilleur
mode = Mode.MUSCLE
batch_size=32
n_layers=1
n_nodes=[25]
activations=[nn.GELU()]
activation_names = ["GELU"]

L1_penalty=0.01
L2_penalty=0.01
learning_rate=0.001
num_epochs=1000 
optimizer=0.0
criterion = ModifiedHuberLoss(delta=0.2, factor=1.0)
p_dropout=0.2
use_batch_norm=True

folder = "datas"
num_folds = 5 # for 80% - 20%
num_try_cross_validation = 10

Hyperparameter_essai1 = ModelHyperparameters(model_name, mode, batch_size, n_layers, n_nodes, activations, activation_names, 
                                             L1_penalty, L2_penalty, learning_rate, num_epochs, criterion, p_dropout, 
                                             use_batch_norm)
print(Hyperparameter_essai1)

# # one model per muscle !
main_superised_learning(Hyperparameter_essai1, q_ranges, folder_name="data_generation_datas_with_dlmt_dq", muscle_name = "PECM2", retrain=True, 
                        file_path=Hyperparameter_essai1.model_name,plot_preparation=False, plot=True, save=True) 
# main_superised_learning(Hyperparameter_essai1, q_ranges, folder_name="datas", muscle_name = "PECM3", retrain=False, 
#                         file_path=Hyperparameter_essai1.model_name,plot_preparation=True, plot=True, save=True) 

# list_simulation, best_hyperparameters_loss, best_hyperparameters_acc = find_best_hyperparameters(Hyperparameter_essai1, q_ranges, "datas", "PECM2")
# all_cross_val_test = try_best_hyperparams_cross_validation(folder_name, list_simulation, num_try_cross_validation , num_folds)

print("")

# # cross_validation("datas/error_part", Hyperparameter_essai1, num_folds)


# -----------------------------------------------------------------

# ------------
q_initial = np.array([0.0 for k in range (8)])
# essai_dlmt_dq(model, q, 0, delta_qi = 1e-4)

# --------------------------
# test

# NE FONCTIONNE QUE POUR PECM2 !
# muscle_index = 0 #(PECM2)
# muscle_selected = "PECM2"
# cylinders_PECM2=[cylinder_T_PECM2]
# q= np.array([(ranges[0] + ranges[-1]) / 2  for ranges in q_ranges])

# compute_lmt(model, q, cylinders_PECM2, muscle_index, plot=False, plot_cadran = False)

# plot_lever_arm(model_biorbd, q_initial, cylinders_PECM2, muscle_selected, "One_cylinder_wrapping_PECM2_T",100)







