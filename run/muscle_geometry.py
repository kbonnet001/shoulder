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
# from neural_networks.main_trainning import test_model_supervised_learning

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

# segment_names
# ['thorax', 'spine', 'clavicle_effector_right', 'clavicle_right', 'scapula_effector_right', 'scapula_right', 'humerus_right', 'ulna_effector_right', 'ulna_right', 'radius_effector_right', 'radius_right', 'hand_right']

# INPUTS :  
# --------

# Datas pour le cylindre (à priori) du thorax pour PECM2 et PECM3 (à partir de deux points)
C_T_PECM2_1 = np.array([0.0187952455, -0.0992656744, 0.0784311931])
C_T_PECM2_2 = np.array([0.0171630409, -0.014154527, 0.0749634019])

# C_T_PECM2_1 = np.array([0.0292629922, -0.1122388526, 0.0788642283])
# C_T_PECM2_2 = np.array([0.0274894968, -0.0197601115, 0.0750962498])


# Datas for cylinder's humerus right (muscle PECM1, PECM2 and PCM3)
# C_H_PECM2_1 = np.array([-0.0452034817, -0.0711088305, 0.175903012])
# C_H_PECM2_2 = np.array([-0.0273048302, -0.0009948723, 0.1717388404])
# C_H_PECM2_1 = np.array([-0.0472718149, -0.0690986252, 0.1764003696])
# C_H_PECM2_2 = np.array([-0.0281573875, 0.0057778626, 0.1719533444])

C_H_PECM2_1 = np.array([-0.0468137093, -0.069205313, 0.1748923225])
C_H_PECM2_2 = np.array([-0.0276992818, 0.0056711748, 0.1704452973])

cylinder_T_PECM2 = Cylinder.from_points(0.025, -1, C_T_PECM2_1, C_T_PECM2_2, "thorax")
print("matrix = ", cylinder_T_PECM2.matrix)

cylinder_H_PECM2 = Cylinder.from_points(0.02, 1, C_H_PECM2_1, C_H_PECM2_2, "humerus_right")

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

# liste des cylindres pour PECM2 (et PECM3)
# une liste de cylindre pour un muscle peut contenir 0, 1 ou 2 cylindres
cylinders_PECM2=[cylinder_T_PECM2, cylinder_H_PECM2]

# q ranges pour PECM2
q_ranges_PECM2 = copy.deepcopy(q_ranges)
# on pose des limites pour eviter les problèmes
q_ranges_PECM2[2][0] = (0.76039816 -0.05) / 2 # de 1/4 à 4/4
q_ranges_PECM2[0][0] =  (0.65449847 -1.04719755 ) / 2 # de 1/4 à 4/4
q_ranges_PECM2[1][0] = -1.8690706375000001  # de 1/2 à 4/4

# TEST --------------
# Pour test, q limites (pour muscle PECM2)
q_test_limite = [[0,0,0],[0,0,0],[0,0,0]]
for k in range (3) :
   q_test_limite[k][0]  = q_ranges_PECM2 [k][0] 
   q_test_limite[k][1]  = (q_ranges_PECM2 [k][0]  + q_ranges_PECM2 [k][1] ) /2
   q_test_limite[k][2]  = q_ranges_PECM2 [k][1] 
   print("q_test_limite = ", q_test_limite)

# ----------------------
muscle_selected = "PECM2"

# test_limit_data_for_learning(muscle_selected,cylinders_PECM2, model, q_ranges,"PECM2_datas00.xlsx", True) 

# data_for_learning (muscle_selected,cylinders_PECM2, model, q_ranges_PECM2,10, "df_PECM2_datas_plusgrand2.xlsx") 
# ----------------------
# test_model_supervised_learning("df_PECM2_datas_15259.xlsx")
# ----------------------
data_for_learning_plot (muscle_selected, cylinders_PECM2, model, q_ranges, 2, 100, plot=False)

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