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

from neural_networks.data_generation import data_for_learning, test_limit_data_for_learning

#################### 
import biorbd
import bioviz

model = biorbd.Model("models/Wu_DeGroote.bioMod")
q = np.zeros((model.nbQ(), ))

# Names of all segment's model 
segment_names = [model.segment(i).name().to_string() for i in range(model.nbSegment())]
humerus_index = segment_names.index("humerus_right") # pour trouver où est humerus_index --» 6

humerus_dof_names = [model.segment(humerus_index).nameDof(i).to_string() for i in range(model.segment(humerus_index).nbQ())]
q_ranges = [[ranges.min(), ranges.max()] for ranges in model.segment(humerus_index).QRanges()]

# INPUTS (Example with PECM2) :  
# -----------------------------

# Datas for cylinder's thorax (muscle PECM2 and PCM3)
C_T_PECM2_1 = np.array([0.0187952455, -0.0992656744, 0.0784311931])
C_T_PECM2_2 = np.array([0.0171630409, -0.014154527, 0.0749634019])

# C_T_PECM2_1 = np.array([0.0292629922, -0.1122388526, 0.0788642283])
# C_T_PECM2_2 = np.array([0.0274894968, -0.0197601115, 0.0750962498])


# Datas for cylinder's humerus right (muscle PECM1, PECM2 and PCM3)
# C_H_PECM2_1 = np.array([-0.0452034817, -0.0711088305, 0.175903012])
# C_H_PECM2_2 = np.array([-0.0273048302, -0.0009948723, 0.1717388404])
C_H_PECM2_1 = np.array([-0.0427634125, -0.0615504057, 0.175335323])
C_H_PECM2_2 = np.array([-0.0072928929, 0.0773974095, 0.167083007])

cylinder_T_PECM2 = Cylinder.from_points(0.025, 1, C_T_PECM2_1, C_T_PECM2_2, "thorax")
print("matrix = ", cylinder_T_PECM2.matrix)

cylinder_H_PECM2 = Cylinder.from_points(0.0169, 1, C_H_PECM2_1, C_H_PECM2_2, "humerus_right")

# cylinder_U = np.array([[ 9.99816470e-01 , 1.91420333e-02, -7.79928088e-04 , 1.79791432e-02],
#                      [ 0.00000000e+00 ,-4.07104880e-02, -9.99170984e-01 ,-5.67101007e-02],
#                      [-1.91579155e-02 , 9.98987607e-01, -4.07030164e-02 , 7.66972975e-02],
#                      [ 0.00000000e+00 , 0.00000000e+00 , 0.00000000e+00 , 1.00000000e+00]])

# cylinder_V = np.array([[-0.79481162  ,0.57873846 ,-0.18302028 ,-0.02082093],
#                         [-0.14524663  ,0.11079776 , 0.98314264 , 0.03127978],
#                         [ 0.58931593 , 0.80782082, -0.00576126 , 0.17126575],
#                         [ 0.      ,	0.    ,  	0.      	,1.    	]])

# cylinder_T_PECM2 = Cylinder.from_matrix(0.025, 1, cylinder_U, "thorax")
# cylinder_H_PECM2 = Cylinder.from_matrix(0.017, 1, cylinder_V, "humerus_right")

# cylinder_T_PECM2.rotate_around_axis(20)
# cylinder_H_PECM2.rotate_around_axis(-10)

# Please, choose a list of cylinder with 0, 1 or 2 cylinders
# List of PECM2's cylinders
cylinders_PECM2=[cylinder_T_PECM2, cylinder_H_PECM2]

# Create q range for PECM2
q_ranges_PECM2 = copy.deepcopy(q_ranges)
# Limite q range to avoid problematic configurations
q_ranges_PECM2[0][0] =  (0.65449847 -1.04719755 ) / 2 # 1/4 -> 4/4
q_ranges_PECM2[1][0] = -1.8690706375000001  # 1/2 -> 4/4
q_ranges_PECM2[2][0] = (0.76039816 -0.05) / 2 # 1/4 -> 4/4
q_ranges_PECM2.append([])
q_ranges_PECM2[3] = [0,2.36]


# TEST --------------
# To test limits of q range with PECM2
q_test_limite = [[0,0,0],[0,0,0],[0,0,0]]
for k in range (3) :
   q_test_limite[k][0]  = q_ranges_PECM2 [k][0] 
   q_test_limite[k][1]  = (q_ranges_PECM2 [k][0]  + q_ranges_PECM2 [k][1] ) /2
   q_test_limite[k][2]  = q_ranges_PECM2 [k][1] 
   print("q_test_limite = ", q_test_limite)
# -------------------

# ----------------------
muscle_selected = "PECM2"

test_limit_data_for_learning(muscle_selected,cylinders_PECM2, model, q_ranges,"PECM2_datas00.xlsx", True) 

# data_for_learning (muscle_selected,cylinders_PECM2, model, q_ranges_PECM2,10, "df_PECM2_datas_plusgrand2.xlsx") 
# ----------------------
# test_model_supervised_learning("df_PECM2_datas_15259.xlsx")
# ----------------------
# data_for_learning (muscle_selected, cylinders_PECM2, model, q_ranges_PECM2, 2, plot=False) 

# pour q0
# origin =  [ 0.0248658 ,-0.0475832,  0.0174664]
# insertion =  [-0.01972176,-0.04075831 , 0.17976807]

# pour q 3
# origin =  [ 0.0248658 ,-0.0475832 , 0.0174664]
# insertion  =  [-0.03443068 , 0.03347737 , 0.18312634]


# Q, G, H, T, Q_G_inactive, H_T_inactive , _= double_cylinder_obstacle_set_algorithm(origin , insertion, cylinder_T_PECM2.matrix, cylinder_T_PECM2.radius, cylinder_H_PECM2.side, cylinder_H_PECM2.matrix, cylinder_H_PECM2.radius, cylinder_H_PECM2.side, np.dot(np.linalg.inv(cylinder_T_PECM2.matrix), cylinder_H_PECM2.matrix), [0,0,0])

# plot_double_cylinder_obstacle(origin, insertion, cylinder_T_PECM2, cylinder_H_PECM2, Q, G, H, T, Q_G_inactive, H_T_inactive)
   

# # Show
# b = bioviz.Viz(loaded_model=model)
# b.set_q(q)
# b.exec()

# exit(0)

#################

# def main():

#    # Provide the length wrapping around a cylinder
#    #  Based on:
#    #  -B.A. Garner and M.G. Pandy, The obstacle-set method for
#    #  representing muscle paths in musculoskeletal models,
#    #  Comput. Methods Biomech. Biomed. Engin. 3 (2000), pp. 1–30.
#    # -----------------------------------------------------------

#    # ------
#    # Inputs
#    # ------
#    # Points
#    P = [0,-4,-2] # origin_point
#    S =[0,4,2] # insersion_point

#    # Points for 1st cylinder
#    center_circle_U = [np.array([-3,0,-2]),np.array([5,0,-2])]
#    radius_U = 1
#    side_U = - 1 # Choose 1 for the right-handed side, -1 for the left-handed side (with respect to the z-axis)

#    # Points for 2nd cylinder
#    center_circle_V = [np.array([0,1.5,-4]),np.array([0,1.5,4])]
#    radius_V = 1
#    side_V = 1 # Choose 1 for the right-handed side, -1 for the left-handed side (with respect to the z-axis)

#    show_details = True
#    single_cylinder = False# with U cylinder
#    double_cylinder = True # with U and V cylinders

#    # Other inputs -----------------------------------------
#    U_origin = (center_circle_U[1] + center_circle_U[0]) / 2
#    cylinder_frame_U = find_cylinder_frame(center_circle_U)
#    matrix_U = np.array([[cylinder_frame_U[0][0], cylinder_frame_U[0][1], cylinder_frame_U[0][2], U_origin[0]],
#                         [cylinder_frame_U[1][0], cylinder_frame_U[1][1], cylinder_frame_U[1][2], U_origin[1]],
#                         [cylinder_frame_U[2][0], cylinder_frame_U[2][1], cylinder_frame_U[2][2], U_origin[2]],
#                         [0, 0, 0, 1]])

#    V_origin = (center_circle_V[1] + center_circle_V[0]) / 2
#    cylinder_frame_V = find_cylinder_frame(center_circle_V)
#    matrix_V = np.array([[cylinder_frame_V[0][0], cylinder_frame_V[0][1], cylinder_frame_V[0][2], V_origin[0]],
#                         [cylinder_frame_V[1][0], cylinder_frame_V[1][1], cylinder_frame_V[1][2], V_origin[1]],
#                         [cylinder_frame_V[2][0], cylinder_frame_V[2][1], cylinder_frame_V[2][2], V_origin[2]],
#                         [0, 0, 0, 1]])

#    # Rotation matrix UV
#    rotation_matrix_UV = np.dot(np.transpose(cylinder_frame_V), cylinder_frame_U)
#    origin_U_in_V_frame = transpose_switch_frame(matrix_U[0:3, 3], matrix_V)

#    matrix_UV = np.dot(np.linalg.inv(matrix_V), matrix_U)

#    if (single_cylinder) :
#     # --------------------------------------
#     # Single cylinder obstacle set algorithm
#     # --------------------------------------
#     v1, v2, obstacle_tangent_point_inactive, segment_length = single_cylinder_obstacle_set_algorithm(P,S, radius_U, side_U, matrix_U)

#     # ----
#     # Plot
#     # ----
#     plot_one_cylinder_obstacle(P,S, center_circle_U, radius_U, v1, v2, obstacle_tangent_point_inactive, segment_length, U_origin, matrix_U)

#     if (show_details) :
#      print("origin_point = ", P)
#      print("final_point = ", S)
#      print("v1 = ", v1)
#      print("v2 = ", v2)
#      print("obstacle_tangent_point_inactive = ",obstacle_tangent_point_inactive)
#      print("segment_length = ", round(segment_length, 2))

#    if (double_cylinder) :
#     # --------------------------------------
#     # Double cylinder obstacle set algorithm
#     # --------------------------------------

#     Q, G, H, T, Q_G_inactive, H_T_inactive, segment_length = double_cylinder_obstacle_set_algorithm(P, S, matrix_U, radius_U, side_U, matrix_V, radius_V, side_V, matrix_UV)

#     # ----
#     # Plot
#     # ----
#     plot_double_cylinder_obstacle(P, S, center_circle_U, center_circle_V, radius_U, radius_V, Q, G, H, T, matrix_U, matrix_V, U_origin, V_origin, Q_G_inactive, H_T_inactive )

#     if (show_details) :
#      print("origin_point = ", P)
#      print("final_point = ", S)
#      print("Q = ", Q)
#      print("G = ", G)
#      print("H = ", H)
#      print("T = ", T)
#      print("Q_G_inactive = ",Q_G_inactive)
#      print("H_T_inactive = ",H_T_inactive)
#      if (Q_G_inactive and H_T_inactive) :
#       print("--> Straight line")
#      elif (Q_G_inactive) :
#       print("--> Single cylinder algo with V")
#      elif (H_T_inactive) :
#       print("--> Single cylinder algo with U")
#      else :
#       print("--> Double cylinder algo with U and V")
#      print("segment_length = ", round(segment_length, 2))

# if __name__ == "__main__":
#    main()