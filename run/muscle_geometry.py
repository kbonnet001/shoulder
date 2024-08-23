import numpy as np
from scipy.linalg import norm
from wrapping.plot_cylinder import *
from wrapping.algorithm import*
from wrapping.Cylinder import Cylinder
from neural_networks.discontinuities import *
import torch.nn as nn
from neural_networks.Loss import *
import random
# from pyorerun import LiveModelAnimation
from neural_networks.file_directory_operations import create_directory, create_and_save_plot
from neural_networks.data_generation import *
from neural_networks.ModelHyperparameters import ModelHyperparameters
from neural_networks.ModelTryHyperparameters import ModelTryHyperparameters
from neural_networks.data_generation_ddl import plot_one_q_variation, data_for_learning_without_discontinuites_ddl, data_generation_muscles, data_for_learning_with_noise, test_limit_data_for_learning, create_all_q_variation_files, plot_all_q_variation, plot_all_q_variation_one
from neural_networks.k_cross_validation import cross_validation, try_best_hyperparams_cross_validation
from neural_networks.functions_data_generation import compute_q_ranges
from neural_networks.muscles_length_jacobian import plot_length_jacobian
from neural_networks.Mode import Mode
from neural_networks.main_trainning import main_supervised_learning, find_best_hyperparameters, plot_results_try_hyperparams
from neural_networks.CSVBatchWriterWithNoise import CSVBatchWriterWithNoise
from neural_networks.Timer import measure_time
from neural_networks.save_model import load_saved_model
from neural_networks.plot_pareto_front import plot_results_try_hyperparams, plot_results_try_hyperparams_comparaison, create_df_from_txt_saved_informations
from neural_networks.analysis_torque_models import compare_model_torque_prediction
from neural_networks.muscle_forces_and_torque import compute_fm_and_torque

#################### 
# Code des tests
import biorbd
import bioviz

import unittest

# Importer les tests
# from wrapping.wrapping_tests.Step1Test import Step1Test
from neural_networks.neural_networks_tests import TestPlotVisualisation
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
cylinders_PECM2=[cylinder_T_PECM2, cylinder_H_PECM2] # list of cylinders for PECM2 (2 wrapping)
cylinders_PECM3=[cylinder_T_PECM3, cylinder_H_PECM3] # list of cylinders for PECM3 (2 wrapping)
cylinders = [cylinders_PECM2, cylinders_PECM3] # list of cylinders PECM2 and PECM3

muscles_selected = ["PECM2", "PECM3"]
# segments_selected = ["thorax", "humerus_right"] 
# -----------------------------------------------------------------

# Visualisation
#--------------

# test_limit_data_for_learning(muscles_selected[0],cylinders_PECM2, model_biorbd, True, True) 

# q_fixed = np.zeros(model_biorbd.nbQ())
# q_fixed[-2] = -1.4311
# plot_one_q_variation(muscles_selected[0], cylinders_PECM2, model_biorbd, q_fixed, 
#                         6, "PECM2", 50, plot_all=False, plot_limit=False, plot_cadran=False)

# create_directory("example_PECM2")
# create_all_q_variation_files(muscles_selected[0], cylinders_PECM2, model_biorbd, q_fixed, "", num_points=100, plot_all=False, 
#                              plot_limit=False, plot_cadran=False, file_path="example_PECM2")
# plot_all_q_variation_one(model_biorbd, q_fixed, 'segment_length', "", file_path="example_PECM2")
# plot_all_q_variation(model_biorbd, q_fixed, 'muscle_force_', model_biorbd.nbMuscles(), "", file_path="example_PECM2")
# plot_all_q_variation(model_biorbd, q_fixed, 'torque_', model_biorbd.nbQ(), "", file_path="example_PECM2")

# Generate datas : 
#----------------
# data_generation_muscles(muscles_selected, cylinders, model_biorbd, 10000, 0, "", num_points = 20, 
#                         plot_cylinder_3D=False, plot_discontinuities = False, plot_cadran = False, plot_graph=False)

# -----------------------------------------------------------------
# Neural network 
num_datas_for_dataset = 25000
with_noise = False

# Try Hyperparams

# data_loaders = prepare_data_from_folder(32, "datas", plot=False)
# print("")

# model_name = "dlmt_dq_f_128_2c"
# mode = Mode.DLMT_DQ_FM
# batch_size = 128
# n_nodes = [[128, 128], [256, 256], [512, 512], [1024, 1024], [2048, 2048]]
# activations = [[nn.GELU(), nn.GELU()]]
# activation_names = [["GELU", "GELU"]]
# L1_penalty = [0.0, 0.1, 0.001]
# L2_penalty = [0.0, 0.1, 0.001]
# learning_rate = [1e-2]
# num_epochs = 1000
# # criterion = ModifiedHuberLoss(delta=0.2, factor=1.0)
# criterion = [
#     # (LogCoshLoss, {'factor': [1.0]}),
#     (ModifiedHuberLoss,  {'delta': [0.2], 'factor': [0.5]}),
#     # (ExponentialLoss, {'alpha': [0.5]}),
#     # (nn.MSELoss, {})
# ]
# p_dropout = [0.0, 0.2]
# use_batch_norm = True

# try_hyperparams = ModelTryHyperparameters(model_name, batch_size, n_nodes, activations, activation_names, 
#                                              L1_penalty, L2_penalty, learning_rate, num_epochs, criterion, p_dropout, 
#                                              use_batch_norm)

# best_hyperparameters_loss \
# = find_best_hyperparameters(try_hyperparams, mode, model_biorbd.nbQ(), model_biorbd.nbSegment(), model_biorbd.nbMuscles(), 
#                             num_datas_for_dataset, "data_generation_via_point_25000", "PECM2", with_noise)

# Hyperparams

model_name="dlmt_dq_f_128_2c/Best_hyperparams" 
mode = Mode.DLMT_DQ_FM
batch_size=128
# n_layers=1
n_nodes=[2048, 2048, 2048]
activations=[nn.GELU(), nn.GELU(),nn.GELU()]
# activations = [nn.Sigmoid()]
activation_names = ["GELU", "GELU", "GELU"]

L1_penalty=0.001
L2_penalty=0.01
learning_rate=0.01
num_epochs=1000  
optimizer=0.0
# criterion = LogCoshLoss(factor=1.8)
criterion = ModifiedHuberLoss(delta=0.2, factor=0.5)
# criterion = nn.MSELoss()
p_dropout=0.2
use_batch_norm=True

hyperparameter = ModelHyperparameters(model_name, batch_size, n_nodes, activations, activation_names, 
                                             L1_penalty, L2_penalty, learning_rate, num_epochs, criterion, p_dropout, 
                                             use_batch_norm)

# main_supervised_learning(hyperparameter, mode, model_biorbd.nbQ(), model_biorbd.nbSegment(), model_biorbd.nbMuscles(), num_datas_for_dataset, 
#                          folder_name="data_generation_via_point_25000", muscle_name = "PECM2", retrain=False, 
#                          file_path=hyperparameter.model_name, with_noise = False, plot_preparation=True, 
#                          plot_loss_acc=True, plot_loader=True, save=True) 

#------------------------------------------------------
# Compare model

# plot_results_try_hyperparams("data_generation_via_point_25000/PECM2/_Model/torque_128_3c/torque_128_3c.CSV",
#                              "execution_time_use_saved_model", "test_acc", "dropout_prob")

# plot_results_try_hyperparams_comparaison(["data_generation_via_point_25000/PECM2/_Model/torque_128_2c/torque_128_2c.CSV", 
#                                           "data_generation_via_point_25000/PECM2/_Model/torque_128_3c/torque_128_3c.CSV", 
#                                           "data_generation_via_point_25000/PECM2/_Model/dlmt_dq_f_128_2c/dlmt_dq_f_128_2c.CSV", 
#                                           "data_generation_via_point_25000/PECM2/_Model/dlmt_dq_f_torque_128_2c/dlmt_dq_f_torque_128_2c.CSV"], 
#                                          "execution_time_use_saved_model", "test_acc", 
#                                          "data_generation_via_point_25000/PECM2/_Model", "num_try")



save_model_paths = ["data_generation_via_point_25000/PECM2/_Model/torque_128_2c/Best_hyperparams", 
                    "data_generation_via_point_25000/PECM2/_Model/dlmt_dq_f_128_2c/Best_hyperparams", 
                    "data_generation_via_point_25000/PECM2/_Model/dlmt_dq_f_torque_128_2c/Best_hyperparams"]
modes = [Mode.TORQUE, Mode.DLMT_DQ_FM, Mode.DLMT_DQ_F_TORQUE]
csv_path_datas = ["data_generation_via_point_25000/PECM2/PECM2.CSV", 
                  "data_generation_via_point_25000/PECM2/PECM2_test.CSV",
                  "data_generation_via_point_25000/PECM2/plot_all_q_variation_/0_clavicle_effector_right_RotX_.CSV",
                  "data_generation_via_point_25000/PECM2/plot_all_q_variation_/1_clavicle_effector_right_RotY_.CSV",
                  "data_generation_via_point_25000/PECM2/plot_all_q_variation_/2_scapula_effector_right_RotX_.CSV",
                  "data_generation_via_point_25000/PECM2/plot_all_q_variation_/3_scapula_effector_right_RotY_.CSV",
                  "data_generation_via_point_25000/PECM2/plot_all_q_variation_/4_scapula_effector_right_RotZ_.CSV",
                  "data_generation_via_point_25000/PECM2/plot_all_q_variation_/5_humerus_right_RotY_.CSV",
                  "data_generation_via_point_25000/PECM2/plot_all_q_variation_/6_humerus_right_RotX_.CSV",
                  "data_generation_via_point_25000/PECM2/plot_all_q_variation_/7_humerus_right_RotY_1_.CSV"]

compare_model_torque_prediction(save_model_paths, modes, model_biorbd.nbQ(), csv_path_datas, 
                                "data_generation_via_point_25000/PECM2/_Model/Comparaison_models")

#-------------------------------------
# cross validation a ameliorer ...

num_folds = 5 # for 80% - 20%
num_try_cross_validation = 10
# cross_validation("data_generation_via_point_25000/PECM2/PECM2.CSV", Hyperparameter_essai1, mode, num_folds, model_biorbd.nbSegment())

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