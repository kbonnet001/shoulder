import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import norm
from wrapping.plot_cylinder import *
from wrapping.algorithm import*
from wrapping.Cylinder import Cylinder
from neural_networks.discontinuities import *
import torch.nn as nn
import torch
from neural_networks.Loss import *
from pyorerun import LiveModelAnimation

from sklearn.model_selection import train_test_split
# from neural_networks.data_preparation import print_informations_environment
# from neural_networks.main_trainning import main_superised_learning

from neural_networks.data_generation import *
from neural_networks.main_trainning import *
from neural_networks.ModelHyperparameters import ModelHyperparameters
from neural_networks.data_generation_ddl import data_for_learning_ddl, plot_one_q_variation, plot_all_q_variation, data_for_learning_without_discontinuites_ddl, data_generation_muscles
from neural_networks.k_cross_validation import cross_validation, try_best_hyperparams_cross_validation
from neural_networks.functions_data_generation import compute_q_ranges

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

model = biorbd.Model("models/Wu_DeGroote.bioMod")

# def compute_q_ranges_segment(model, segment_selected) : 
#     # segment_names
#     # ['thorax', 'spine', 'clavicle_effector_right', 'clavicle_right', 'scapula_effector_right', 'scapula_right', 
#     # 'humerus_right', 'ulna_effector_right', 'ulna_right', 'radius_effector_right', 'radius_right', 'hand_right']
    
#     segment_names = [model.segment(i).name().to_string() for i in range(model.nbSegment())]
#     humerus_index = segment_names.index(segment_selected) 

#     # humerus_dof_names = [model.segment(humerus_index).nameDof(i).to_string() for i in 
#     #                     range(model.segment(humerus_index).nbQ())]
    
#     q_ranges = [[ranges.min(), ranges.max()] for ranges in model.segment(humerus_index).QRanges()]
#     return q_ranges


# q_ranges = compute_q_ranges_segment(model, "humerus_right")
# q_ranges.append([0.05,  2.356194490192345])

q_ranges, _ = compute_q_ranges(model)


# INPUTS :  
# --------

# Datas pour le cylindre (à priori) du thorax pour PECM2 et PECM3 (à partir de deux points)
C_T_PECM2_1 = np.array([0.0183539873, -0.0762563082, 0.0774936934])
C_T_PECM2_2 = np.array([0.0171218365, -0.0120059285, 0.0748758588])

# Datas for cylinder's humerus right (muscle PECM1, PECM2 and PCM3)
# C_H_PECM2_1 = np.array([-0.0468137093, -0.069205313, 0.1748923225]) # 0.02 5000_2 
# C_H_PECM2_2 = np.array([-0.0276992818, 0.0056711748, 0.1704452973])

# C_H_PECM2_1 = np.array([-0.0549820662, -0.0634122725, 0.1813890163])
# C_H_PECM2_2 = np.array([-0.0398385092, -0.00409078, 0.1778658253]) # 0.0285

# C_H_PECM2_1 = np.array([-0.0414664438, -0.0640982989, 0.175861232])
# C_H_PECM2_2 = np.array([-0.0278003171, -0.0105643089, 0.1726817693]) #0.016

# C_H_PECM2_1 = np.array([-0.0418935136, -0.0638634585, 0.1779761963])
# C_H_PECM2_2 = np.array([-0.028227486, -0.0103298566, 0.1747967566]) # r = 0.016 5000_6 avec q_ranges_PECM2

#----------------------------------------
# re test demade de descendre le cylindre
# C_H_PECM2_1 = np.array([-0.0425358482, -0.0644412567, 0.1813890163])
# C_H_PECM2_2 = np.array([-0.0284186112, -0.0091401431, 0.1781046015]) # r = 0.016 

# C_H_PECM2_1 = np.array([-0.0425358482, -0.0644412567, 0.1830806943])
# C_H_PECM2_2 = np.array([-0.028370612, -0.008952117, 0.1797851123]) # r = 0.016 plus de probleme pour q fixe midle range et q0 varie

# C_H_PECM2_1 = np.array([-0.0424215591, -0.0643620752, 0.1829936853])
# C_H_PECM2_2 = np.array([-0.0283105583, -0.0090853905, 0.1797107213]) # r = 0.016 5000_ 3 et 4

# C_H_PECM2_1 = np.array([-0.0409602858, -0.0641361296, 0.1736150806])
# C_H_PECM2_2 = np.array([-0.0274015472, -0.0110228084, 0.170460602]) # r = 0.016 descendre

# re essai des cadrans
# C_H_PECM2_1 = np.array([-0.03848864,-0.05134674,0.1740978])
# C_H_PECM2_2 = np.array([-0.0137964 ,  0.04537976,  0.16835304]) # r = 0.016 fait à partir de insertion en ref segment humerus local

C_H_PECM2_1 = np.array([-0.0504468139, -0.0612220954, 0.1875298764])
C_H_PECM2_2 = np.array([-0.0367284615, -0.0074835226, 0.1843382632]) #le mieux avec 0.025 0.0243 vrai 0.0255913399

# Presque la fin j'espere ahhhh
# C_H_PECM2_1 = np.array([-0.0439209708, -0.0622294269, 0.1772302577])
# C_H_PECM2_2 = np.array([-0.030809488, -0.0108681304, 0.1741798345]) # pas bien, ne pas faire des cylinre avec r trop petit !

#################################################################################################
cylinder_T_PECM2 = Cylinder.from_points(0.025, -1, C_T_PECM2_2, C_T_PECM2_1, False, "thorax", "PECM2")
cylinder_H_PECM2 = Cylinder.from_points(0.0255913399, 1, C_H_PECM2_2, C_H_PECM2_1, True, "humerus_right", "PECM2")

C_T_PECM3_1 = np.array([-0.0504468139, -0.0612220954, 0.1875298764])
C_T_PECM3_2 = np.array([0.0182587352, -0.0712893992, 0.0772913203])

C_H_PECM3_1 = np.array([-0.0468137093,-0.069205313,0.1748923225])
C_H_PECM3_2 = np.array([-0.0367284615, -0.0074835226, 0.1843382632])

cylinder_T_PECM3 = Cylinder.from_points(0.025, -1, C_T_PECM3_1, C_T_PECM3_2, False, "thorax","PECM3")
cylinder_H_PECM3 = Cylinder.from_points(0.0202946443, 1, C_H_PECM3_1, C_H_PECM3_2, True, "humerus_right", "PECM3")

# cylinder_H_PECM2.rotate_around_axis(-45)

cylinders_PECM2=[cylinder_T_PECM2, cylinder_H_PECM2]
cylinders_PECM3=[cylinder_T_PECM3, cylinder_H_PECM3]

cylinders = [cylinders_PECM2, cylinders_PECM3]

muscles_selected = ["PECM2", "PECM3"]
segments_selected = ["thorax", "humerus_right"] # pour le moment, on change rien

# test_limit_data_for_learning(muscles_selected[0],cylinders_PECM2, model, q_ranges, True, False) 

# data_for_learning (muscles_selected[0],cylinders_PECM2, model, q_ranges, 5000, "df_PECM2_datas_without_error_partfdsadaf_5000.xlsx", False, False) 

# data_for_learning_ddl (muscles_selected[0], cylinders_PECM2, model, 10, "rgtrsfdd.xlsx", data_without_error = True, plot=False, plot_cadran = False)
   
# ----------------------
# train_model_supervised_learning("df_PECM2_datas_5000_more.xlsx")
# print_informations_environment()
# main_superised_learning("df_PECM2_datas_25000.xlsx", True, "model_weights.pth")
# ----------------------

q_fixed = np.array([(ranges[0] + ranges[-1]) / 2  for ranges in q_ranges])
# q_fixed = np.array([0.0,0.0,0.0,0.0])
# q_fixed = np.array([(ranges[1]) for ranges in q_ranges])
# q_fixed = np.array([q_ranges[0][0], q_ranges[1][1], q_ranges[2][0], 0.0])
# q_fixed = np.array([(ranges[0] + ranges[-1]) / 2  for ranges in q_ranges_PECM2])

# plot_one_q_variation(muscles_selected[0], cylinders_PECM2, model, q_fixed, 
#                         1, "essai2", 100, plot_all=False, plot_limit=False)

# plot_all_q_variation(muscles_selected[0], cylinders_PECM2, model, q_fixed, "ngafsdhgf", num_points = 100, 
#                      plot_all = False, plot_limit = False, plot_cadran=False)

# data_for_learning_without_discontinuites_ddl(muscles_selected[1], cylinders[1], model, 5000, "data_testfraszfdsc", num_points = 50, plot_cylinder_3D=False, plot_discontinuities = False, plot_cadran = False, plot_graph=True)

data_generation_muscles(muscles_selected, cylinders, model, 5000, "data_more_ddl", num_points = 50, plot_cylinder_3D=False, plot_discontinuities = False, plot_cadran = False, plot_graph=False)

# data_for_learning_without_discontinuites(muscles_selected, cylinders, model, q_ranges, 5000, 
#                 "df_PECM3_datas_without_error_part_5000.xlsx", num_points = 50, plot_discontinuities = False, 
#                 plot=False, plot_cadran = False)
   

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


# Show biorbd
# q = np.zeros((model.nbQ(), ))
# b = bioviz.Viz(loaded_model=model)
# b.set_q(q)
# b.exec()

# exit(0)

# pour voir pyorerun
# model_path = "/home/lim/Documents/kloe/shoulder/run/models/Wu_DeGroote.bioMod"
# animation = LiveModelAnimation(model_path, with_q_charts=True)
# animation.rerun()


#################
p0 = np.array([1.0, 1.0])
p1 = np.array([2.0, 2.0])
p2 = np.array([3.0, 3.0])

# find_discontinute(p0, p1, p2)

# data_loaders = prepare_data_from_folder(32, "datas", plot=False)
# print("")

# model_name = "H_essai_1"
# batch_size = 32
# n_layers = [1]
# n_nodes = [[8], [10], [15], [20], [25], [30]]
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


model_name="essai"
batch_size=128
n_layers=1
n_nodes=[50]
activations=[nn.GELU()]
activation_names = ["GELU"]
L1_penalty=0.01
L2_penalty=0.01
learning_rate=0.00001
num_epochs=1000 
optimizer=0.0
criterion = ModifiedHuberLoss(delta=2.0, factor=2.0)
p_dropout=0.2
use_batch_norm=True

folder_name = "datas/error_part"
num_folds = 5 # for 80% - 20%
num_try_cross_validation = 10

Hyperparameter_essai1 = ModelHyperparameters(model_name, batch_size, n_layers, n_nodes, activations, activation_names, L1_penalty, 
                              L2_penalty, learning_rate, num_epochs, criterion, p_dropout, use_batch_norm)

# main_superised_learning(Hyperparameter_essai1, "data_generation_data_more_ddl", True, "essai_bestparameter_1",False, True, True) 

# list_simulation, best_hyperparameters_loss, best_hyperparameters_acc = find_best_hyperparameters(Hyperparameter_essai1, folder_name)
# all_cross_val_test = try_best_hyperparams_cross_validation(folder_name, list_simulation, num_try_cross_validation , num_folds)

print("")

# # cross_validation("datas/error_part", Hyperparameter_essai1, num_folds)




