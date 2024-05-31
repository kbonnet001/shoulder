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

from neural_networks.prepare_data import data_for_learning

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

# Datas for cylinder's humerus right (muscle PECM1, PECM2 and PCM3)
C_H_PECM2_1 = np.array([-0.0452034817, -0.0711088305, 0.175903012])
C_H_PECM2_2 = np.array([-0.0273048302, -0.0009948723, 0.1717388404])

cylinder_T_PECM2 = Cylinder(0.025, -1, C_T_PECM2_1, C_T_PECM2_2, "thorax")
cylinder_H_PECM2 = Cylinder(0.0175, 1, C_H_PECM2_1, C_H_PECM2_2, "humerus_right")

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

data_for_learning (muscle_selected,cylinders_PECM2, model, q_ranges_PECM2,500, "df_PECM2_datas2.xlsx") 
# ----------------------

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