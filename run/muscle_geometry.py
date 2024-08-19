import numpy as np
from scipy.linalg import norm
from wrapping.plot_cylinder import *
from wrapping.algorithm import*
from wrapping.Cylinder import Cylinder
from neural_networks.discontinuities import *
import torch.nn as nn
from neural_networks.Loss import *
# from pyorerun import LiveModelAnimation

from neural_networks.data_generation import *
from neural_networks.ModelHyperparameters import ModelHyperparameters
from neural_networks.ModelTryHyperparameters import ModelTryHyperparameters
from neural_networks.data_generation_ddl import plot_one_q_variation, data_for_learning_without_discontinuites_ddl, data_generation_muscles, data_for_learning_with_noise, test_limit_data_for_learning
from neural_networks.k_cross_validation import cross_validation, try_best_hyperparams_cross_validation
from neural_networks.functions_data_generation import compute_q_ranges
from neural_networks.muscles_length_jacobian import plot_length_jacobian
from neural_networks.Mode import Mode
from neural_networks.main_trainning import main_supervised_learning, find_best_hyperparameters, plot_results_try_hyperparams
from neural_networks.CSVBatchWriterWithNoise import CSVBatchWriterWithNoise
from neural_networks.Timer import measure_time
from neural_networks.save_model import load_saved_model
from neural_networks.plot_pareto_front import plot_results_try_hyperparams, plot_results_try_hyperparams_comparaison, create_df_from_txt_saved_informations

#################### 
# Code des tests
import biorbd
# import bioviz

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

# test_limit_data_for_learning(muscles_selected[0],cylinders_PECM2, model_biorbd, q_ranges, True, True) 

# data_for_learning (muscles_selected[0],cylinders_PECM2, model_biorbd, q_ranges, 5000, "df_PECM2_datas_without_error_partfdsadaf_5000.csv", False, False) 

# data_for_learning_ddl (muscles_selected[0], cylinders_PECM2, model_biorbd, 10, "rgtrsfdd.csv", data_without_error = True, plot=False, plot_cadran = False)

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
# data_generation_muscles(muscles_selected, cylinders, model_biorbd, 10, 0, "dhhfgfhg", num_points = 20, 
#                         plot_cylinder_3D=False, plot_discontinuities = False, plot_cadran = False, plot_graph=False)

   
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

model_name = "dlmt_dq_f_torque_64_2c"
mode = Mode.DLMT_DQ_F_TORQUE
batch_size = 64
n_nodes = [[512, 512]]
activations = [[nn.GELU(), nn.GELU()]]
activation_names = [["GELU", "GELU"]]
L1_penalty = [0.0, 0.1, 0.001]
L2_penalty = [0.0, 0.1, 0.001]
learning_rate = [1e-2]
num_epochs = 1000
# criterion = ModifiedHuberLoss(delta=0.2, factor=1.0)
criterion = [
    # (LogCoshLoss, {'factor': [1.0]}),
    (ModifiedHuberLoss,  {'delta': [0.2], 'factor': [0.5]}),
    # (ExponentialLoss, {'alpha': [0.5]}),
    # (nn.MSELoss, {})
]
p_dropout = [0.0, 0.2, 0.5]
use_batch_norm = True


# model_name="essai_muscle_train"
# mode = Mode.MUSCLE
# batch_size=64
# # n_layers=1
# n_nodes=[25]
# activations=[nn.GELU()]
# activation_names = ["GELU"]

# model_name="hfhjsdf" 
# mode = Mode.TORQUE
# batch_size=64
# # n_layers=1
# n_nodes=[256, 256]
# activations=[nn.GELU(), nn.GELU()]
# # activations = [nn.Sigmoid()]

# activation_names = ["GELU", "GELU"]

# L1_penalty=0.01
# L2_penalty=0.01
# learning_rate=0.01
# num_epochs=1000 
# optimizer=0.0
# # criterion = LogCoshLoss(factor=1.8)
# criterion = ModifiedHuberLoss(delta=0.1, factor=0.2)
# # criterion = nn.MSELoss()
# p_dropout=0.2
# use_batch_norm=True

num_datas_for_dataset = 10
folder = "datas"
num_folds = 5 # for 80% - 20%
num_try_cross_validation = 10
with_noise = False

# Hyperparameter_essai1 = ModelHyperparameters(model_name, batch_size, n_nodes, activations, activation_names, 
#                                              L1_penalty, L2_penalty, learning_rate, num_epochs, criterion, p_dropout, 
#                                              use_batch_norm)
Hyperparameter_essai1 = ModelTryHyperparameters(model_name, batch_size, n_nodes, activations, activation_names, 
                                             L1_penalty, L2_penalty, learning_rate, num_epochs, criterion, p_dropout, 
                                             use_batch_norm)
print(Hyperparameter_essai1)

# test_limit_data_for_learning ("PECM2", cylinders_PECM2, model_biorbd, plot=True, plot_cadran=False)

# one model per muscle !

# main_supervised_learning(Hyperparameter_essai1, mode, model_biorbd.nbQ(), model_biorbd.nbSegment(), num_datas_for_dataset, 
#                          folder_name="data_generation_via_point", muscle_name = "PECM2", retrain=True, 
#                          file_path=Hyperparameter_essai1.model_name, with_noise = False, plot_preparation=False, 
#                          plot=True, save=True) 


best_hyperparameters_loss \
= find_best_hyperparameters(Hyperparameter_essai1, mode, model_biorbd.nbQ(), model_biorbd.nbSegment(), 
                            num_datas_for_dataset, "data_generation_via_point", "PECM2", with_noise)

# plot_results_try_hyperparams("data_generation_via_point/PECM2/_Model/dlmt_dq_f_torque_64_2c/dlmt_dq_f_torque_64_2c.CSV",
#                                  "execution_time_use_saved_model", "val_loss", 'L2_penalty')

# plot_results_try_hyperparams_comparaison(["data_generation_via_point/PECM2/_Model/torque_64_1c/torque_64_1c.CSV", 
#                                           "data_generation_via_point/PECM2/_Model/torque_64_2c/torque_64_2c.CSV", 
#                                           "data_generation_via_point/PECM2/_Model/dlmt_dq_f_torque_64_2c/dlmt_dq_f_torque_64_2c.CSV"], 
#                                          "execution_time_use_saved_model", "val_acc", "data_generation_datas_with_tau/PECM2/_Model", "num_try")

# plot_results_try_hyperparams_comparaison(["data_generation_via_point/PECM2/_Model/torque_64_2c/torque_64_2c.CSV", 
#                                           "data_generation_via_point/PECM2/_Model/torque_64_1c/torque_64_1c.CSV"], 
#                                          "execution_time_use_saved_model", "val_loss", 
#                                          "data_generation_via_point/PECM2/_Model", 'num_try')

# create_df_from_txt_saved_informations("data_generation_datas_with_tau/PECM2/_Model/train_torque_1c_64/train_torque_1c_64.csv") 

# all_cross_val_test = try_best_hyperparams_cross_validation(folder_name, list_simulation, num_try_cross_validation , num_folds)

print("")

# cross_validation("data_generation_via_point/PECM2", Hyperparameter_essai1, mode, num_folds, model_biorbd.nbSegment())


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

# plot_length_jacobian(model_biorbd, q_initial, cylinders_PECM2, muscle_selected, "One_cylinder_wrapping_PECM2_T",100)

# from wrapping.muscle_forces_and_torque import compute_muscle_force_origin_insertion_nul, compute_torque
# from wrapping.muscles_length_jacobian import compute_dlmt_dq
# # test_muscle_force()

# model_one_muscle = m = biorbd.Model("models/oneMuscle.bioMod")

# lmt1 = 0.242617769697736
# q1 = np.array([-0.0786849353613072, 
#      0.0714227230995303, 
#      -0.368636743434153, 
#      0.480812853033726, 
#      0.0564779278353358, 
#      -1.0471975511966, 
#      -2.38720333455028, 
#      0.88080460882924])

# dlmt_dq1 = compute_dlmt_dq(model_biorbd, q_ranges, q1, cylinders_PECM2, 0)
# f1 = compute_muscle_force_origin_insertion_nul(model_one_muscle, 0, lmt1)

# lmt2 = 0.139061503308524
# q2 = np.array([-0.00533827452890219, 
#      -0.148900513226347, 
#      0.0964274708639019, 
#      -0.0126420791755437, 
#      0.0695855116103103, 
#      2.1860248881229, 
#     -2.15548447226009, 
#      1.33380777124826])

# dlmt_dq2 = compute_dlmt_dq(model_biorbd, q_ranges, q2, cylinders_PECM2, 0)
# f2 = compute_muscle_force_origin_insertion_nul(model_one_muscle, 0, lmt2)

# # lmt3 = 0.268844328998133
# q3 = np.array([-0.0546715244277362, 
#      0.21367517662627, 
#      -0.361552222464051, 
#      0.440176189008029, 
#      0.0606371961599653, 
#      -1.0471975511966, 
#     -2.21153422915413, 
#      1.43189727009164])

# dlmt_dq3 = compute_dlmt_dq(model_biorbd, q_ranges, q3, cylinders_PECM2, 0)
# f3 = compute_muscle_force_origin_insertion_nul(model_one_muscle, 0, lmt3)

# tau1 = compute_torque(np.array(dlmt_dq1), f1)
# print("tau = ", tau1)
# tau2 = compute_torque(np.array(dlmt_dq2), f2)
# print("tau = ", tau2)
# tau3 = compute_torque(np.array(dlmt_dq3), f3)
# print("tau = ", tau3)


# comparaison activation ----------------------

# np.random.seed(42)
# q = np.random.rand(model_biorbd.nbQ())
# qdot = np.random.rand(model_biorbd.nbQ())
# state_set = model_biorbd.stateSet()
# for state in state_set:
#     state.setActivation(1)
# print(model_biorbd.muscleForces(state_set, q, qdot).to_array() / 2)
# print(model_biorbd.muscularJointTorque(state_set, q, qdot).to_array() / 2)
# state_set = model_biorbd.stateSet()
# for state in state_set:
#     state.setActivation(0.5)
# print(model_biorbd.muscleForces(state_set, q, qdot).to_array())
# print(model_biorbd.muscularJointTorque(state_set, q, qdot).to_array())


# -----------------------------------
# file_path_model = 'data_generation_datas_with_dlmt_dq/PECM2/_Model/msucle_for_casadi'
# input_shape = 8

# casadi_model_test = load_model_to_casadi(file_path_model, input_shape)

# input_data = ""
# output_data = casadi_model_test(input_data)


# Instantiate the model
# from neural_networks.casadi import casadi_test
# file_path_model = 'data_generation_datas_with_dlmt_dq/PECM2/_Model/msucle_for_casadi'
# model = load_saved_model(file_path_model)

# # Define the input shape
# input_shape = (1,8)
# q1 = np.array([-0.0786849353613072, 
#      0.0714227230995303, 
#      -0.368636743434153, 
#      0.480812853033726, 
#      0.0564779278353358, 
#      -1.0471975511966, 
#      -2.38720333455028, 
#      0.88080460882924])

# casadi_test(model, q1, len(q1))

# Convert the PyTorch model to a CasADi function
# casadi_model = pytorch_to_casadi(model, input_shape)


# file_path_model = 'data_generation_datas_with_dlmt_dq/PECM2/_Model/msucle_for_casadi'
# model = load_saved_model(file_path_model)

# # Extraire les poids et les biais
# weights, biases = get_weights_biases(model)

# # Définir la fonction analytique dans CasADi
# input_size = 8
# output_size = 1
# activations = ['GELU']
# model_func = pytorch_to_casadi(weights, biases, activations, input_size, output_size)



############
# EXAMPLE 
############

# from neural_networks.save_model import main_function_model

# file_path = 'data_generation_datas_with_dlmt_dq/PECM2/_Model/torque_train_1_couche_8192'
# q1 = [-0.0786849353613072, 
#      0.0714227230995303, 
#      -0.368636743434153, 
#      0.480812853033726, 
#      0.0564779278353358, 
#      -1.0471975511966, 
#      -2.38720333455028, 
#      0.88080460882924]

# main_function_model(file_path, q1)