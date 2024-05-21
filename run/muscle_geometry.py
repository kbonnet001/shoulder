import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import norm
from wrapping.plot_cylinder import *
from wrapping.algorithm import*
from wrapping.step_1 import find_cylinder_frame

# ####################
# import biorbd
# import bioviz

# # Pseudo code : 
# # On choisit le muscle que l’on veut 
# # Ce muscle a un point d,origin et un point d’insertion, P et S, que l'on récupère
# # Pour ce muscle, il existe 1 ou 2 cylindres avec lequel il va falloir faire un wrapping, on les nomme dans l’ordre U et V
# # On va ensuite faire une boucle for pour récupérer ce dont on a besoin 
# # Un q radom définit en fonction de ses bornes min et max
# # Le P et S correspondant à ce q
# # La longueur du chemin de passage
# # On va mettre tout cela dans une matrice tel que on est

# # essai : 

# # Datas pour le cylindre (à piori) de l'humerus pour PECM2 et PECM3 (en coordonnes frame de ref)
# C_H_PECM_1 = np.array([-0.0452034817, -0.0711088305, 0.175903012])
# C_H_PECM_2 = np.array([-0.0273048302, -0.0009948723, 0.1717388404])
# center_circle_H_PECM = [C_H_PECM_2, C_H_PECM_1]
# # Other inputs --------------------------
# H_PECM_origin = (center_circle_H_PECM[1] + center_circle_H_PECM[0]) / 2
# cylinder_frame_H_PECM = find_cylinder_frame(center_circle_H_PECM)

# # se resume à 
# cylinder_H_PECM_ = np. array([[ 0.96903113, -0.24650421,  0.01464025, -0.03625416],
#        [-0.        ,  0.05928701,  0.99824098, -0.03605185],
#        [-0.24693858, -0.96732659,  0.05745096, 0.17382093],
#        [0, 0, 0, 1]])
# radius_H_PECM = 0.025
# side_H_PECM = 1

# T_PECM_origin = np.array([0.0179791432, -0.0567101007, 0.0766972975])
# C_T_PECM_1 = np.array([0.0187952455, -0.0992656744, 0.0784311931])
# C_T_PECM_2 = np.array([0.0171630409, -0.014154527, 0.0749634019])

# model = biorbd.Model("models/Wu_DeGroote.bioMod")
# q = np.zeros((model.nbQ(), ))
# # On choisit le muscle que l’on veut 
# # Pour le moment, on va faire qu'avec pec1 et pec2 (ont leurs cylindres en commun)


# # essai 1

# gcs_ulna_frame =  [[ 0.96734723,-0.24253723, 0.07339071 ],
#  [0.24692246,  0.96726508,  -0.05744761 ],
#  [-0.05693821, 0.07273017,  0.99580371 ]]
# gcs_ulna_vect = np.array([-0.09124765, -0.28625132, 0.1761075  ])

# H_PECM_origin_humerus= transpose_switch_frame(H_PECM_origin, gcs_ulna_frame, [0,0,0] - gcs_ulna_vect )
# C_H_PECM_1_humerus= transpose_switch_frame(C_H_PECM_1, gcs_ulna_frame, [0,0,0] - gcs_ulna_vect )
# C_H_PECM_1_humerus= transpose_switch_frame(C_H_PECM_2, gcs_ulna_frame, [0,0,0] - gcs_ulna_vect )

# cylinder_H_humerus_frame = find_cylinder_frame([np.array(C_H_PECM_1), np.array(C_H_PECM_2)])

# print("H_origin_humerus = ", H_PECM_origin_humerus)
# print("cylinder_H_humerus_frame = ", cylinder_H_humerus_frame)
# # def data_for_learning ( muscle choisi ?)

# # boucle de random q 
# segment_names = [model.segment(i).name().to_string() for i in range(model.nbSegment())]
# humerus_index = segment_names.index("humerus_right")
# print("segment_names =", segment_names, "\n")

# humerus_dof_names = [model.segment(humerus_index).nameDof(i).to_string() for i in range(model.segment(humerus_index).nbQ())]
# q_ranges = [[ranges.min(), ranges.max()] for ranges in model.segment(humerus_index).QRanges()]

# # faire un nouveau q : 


# # Ce muscle a un point d,origin et un point d’insertion, P et S, que l'on récupère
# # à chaque fois que l'on change de q, il faut remettre à jour, on recupere P et S
# model.updateMuscles(q)
# mus = model.muscle(0) # de 0 à 15 inclus donc 16 muscles au total
# origin_0 = mus.musclesPointsInGlobal(model, q)[0].to_array()
# insertion_0 = mus.musclesPointsInGlobal(model, q)[-1].to_array()

# # Pour ce muscle, il existe 1 ou 2 cylindres avec lequel il va falloir faire un wrapping, on les nomme dans l’ordre U et V

# ## code donnee, ne pas effacer 

# model = biorbd.Model("models/Wu_DeGroote.bioMod")
# q = np.zeros((model.nbQ(), ))
# q[1] = -2

# print("q = ", q, "\n")

# segment_names = [model.segment(i).name().to_string() for i in range(model.nbSegment())]
# humerus_index = segment_names.index("humerus_right")
# thorax_index = segment_names.index("thorax")
# ulna_right_index = segment_names.index("ulna_right")
# print("segment_names =", segment_names, "\n")
# print("ulna_right_index = ", ulna_right_index )


# humerus_dof_names = [model.segment(humerus_index).nameDof(i).to_string() for i in range(model.segment(humerus_index).nbQ())]
# q_ranges = [[ranges.min(), ranges.max()] for ranges in model.segment(humerus_index).QRanges()]
# print("humerus_dof_names =", humerus_dof_names, "\n")
# print("q_ranges = ", q_ranges, "\n")

# q[1] = 0
# gcs = [gcs.to_array() for gcs in model.allGlobalJCS(q)]
# gcs_humerus = gcs[humerus_index] # le ref de l'humerus
# gcs_thorax = gcs[thorax_index] # le ref de base 
# gcs_ulna_right = gcs[ulna_right_index] # le ref de ulna

# print("gcs_humerus = ", gcs_humerus)
# # Cylinder pose
# model.updateMuscles(q)
# mus = model.muscle(0) # de 0 à 15 inclus donc 16 muscles au total
# origin_0 = mus.musclesPointsInGlobal(model, q)[0].to_array()
# insertion_0 = mus.musclesPointsInGlobal(model, q)[-1].to_array()

# print("origin_0 = ", origin_0)
# print("insertion_0 = ", insertion_0)

# # q[1] = 0
# # gcs = [gcs.to_array() for gcs in model.allGlobalJCS(q)]
# # gcs_humerus = gcs[humerus_index]
# # # Cylinder pose
# # model.updateMuscles(q)
# # mus = model.muscle(0)
# # origin_1 = mus.musclesPointsInGlobal(model, q)[0].to_array()
# # insertion_1 = mus.musclesPointsInGlobal(model, q)[-1].to_array()


# # transpose => [R_T, -R_T @ t; 0 0 0 1]

# # Show
# b = bioviz.Viz(loaded_model=model)
# b.set_q(q)
# b.exec()

# exit(0)
# #################

def main():

   # Provide the length wrapping around a cylinder
   #  Based on:
   #  -B.A. Garner and M.G. Pandy, The obstacle-set method for
   #  representing muscle paths in musculoskeletal models,
   #  Comput. Methods Biomech. Biomed. Engin. 3 (2000), pp. 1–30.
   # -----------------------------------------------------------

   # ------
   # Inputs
   # ------
   # Points
   P = [0,-4,-2] # origin_point
   S =[0,4,2] # insersion_point

   # Points for 1st cylinder
   center_circle_U = [np.array([-3,0,-2]),np.array([5,0,-2])]
   radius_U = 1
   side_U = - 1 # Choose 1 for the right-handed side, -1 for the left-handed side (with respect to the z-axis)

   # Points for 2nd cylinder
   center_circle_V = [np.array([0,1.5,-4]),np.array([0,1.5,4])]
   radius_V = 1
   side_V = 1 # Choose 1 for the right-handed side, -1 for the left-handed side (with respect to the z-axis)

   show_details = True
   single_cylinder = False# with U cylinder
   double_cylinder = True # with U and V cylinders

   # Other inputs -----------------------------------------
   U_origin = (center_circle_U[1] + center_circle_U[0]) / 2
   cylinder_frame_U = find_cylinder_frame(center_circle_U)
   matrix_U = np.array([[cylinder_frame_U[0][0], cylinder_frame_U[0][1], cylinder_frame_U[0][2], U_origin[0]],
                        [cylinder_frame_U[1][0], cylinder_frame_U[1][1], cylinder_frame_U[1][2], U_origin[1]],
                        [cylinder_frame_U[2][0], cylinder_frame_U[2][1], cylinder_frame_U[2][2], U_origin[2]],
                        [0, 0, 0, 1]])

   V_origin = (center_circle_V[1] + center_circle_V[0]) / 2
   cylinder_frame_V = find_cylinder_frame(center_circle_V)
   matrix_V = np.array([[cylinder_frame_V[0][0], cylinder_frame_V[0][1], cylinder_frame_V[0][2], V_origin[0]],
                        [cylinder_frame_V[1][0], cylinder_frame_V[1][1], cylinder_frame_V[1][2], V_origin[1]],
                        [cylinder_frame_V[2][0], cylinder_frame_V[2][1], cylinder_frame_V[2][2], V_origin[2]],
                        [0, 0, 0, 1]])

   # Rotation matrix UV
   rotation_matrix_UV = np.dot(np.transpose(cylinder_frame_V), cylinder_frame_U)
   origin_U_in_V_frame = transpose_switch_frame(matrix_U[0:3, 3], matrix_V)

   matrix_UV = np.array([[rotation_matrix_UV[0][0], rotation_matrix_UV[0][1], rotation_matrix_UV[0][2], origin_U_in_V_frame[0]],
                        [rotation_matrix_UV[1][0], rotation_matrix_UV[1][1], rotation_matrix_UV[1][2], origin_U_in_V_frame[1]],
                        [rotation_matrix_UV[2][0], rotation_matrix_UV[2][1], rotation_matrix_UV[2][2], origin_U_in_V_frame[2]],
                        [0, 0, 0, 1]])

   if (single_cylinder) :
    # --------------------------------------
    # Single cylinder obstacle set algorithm
    # --------------------------------------
    v1, v2, obstacle_tangent_point_inactive, segment_length = single_cylinder_obstacle_set_algorithm(P,S, radius_U, side_U, matrix_U)

    # ----
    # Plot
    # ----
    plot_one_cylinder_obstacle(P,S, center_circle_U, radius_U, v1, v2, obstacle_tangent_point_inactive, segment_length, U_origin, matrix_U)

    if (show_details) :
     print("origin_point = ", P)
     print("final_point = ", S)
     print("v1 = ", v1)
     print("v2 = ", v2)
     print("obstacle_tangent_point_inactive = ",obstacle_tangent_point_inactive)
     print("segment_length = ", round(segment_length, 2))

   if (double_cylinder) :
    # --------------------------------------
    # Double cylinder obstacle set algorithm
    # --------------------------------------

    Q, G, H, T, Q_G_inactive, H_T_inactive, segment_length = double_cylinder_obstacle_set_algorithm(P, S, matrix_U, radius_U, side_U, matrix_V, radius_V, side_V, matrix_UV)

    # ----
    # Plot
    # ----
    plot_double_cylinder_obstacle(P, S, center_circle_U, center_circle_V, radius_U, radius_V, Q, G, H, T, matrix_U, matrix_V, U_origin, V_origin, Q_G_inactive, H_T_inactive )

    if (show_details) :
     print("origin_point = ", P)
     print("final_point = ", S)
     print("Q = ", Q)
     print("G = ", G)
     print("H = ", H)
     print("T = ", T)
     print("Q_G_inactive = ",Q_G_inactive)
     print("H_T_inactive = ",H_T_inactive)
     if (Q_G_inactive and H_T_inactive) :
      print("--> Straight line")
     elif (Q_G_inactive) :
      print("--> Single cylinder algo with V")
     elif (H_T_inactive) :
      print("--> Single cylinder algo with U")
     else :
      print("--> Double cylinder algo with U and V")
     print("segment_length = ", round(segment_length, 2))

if __name__ == "__main__":
   main()