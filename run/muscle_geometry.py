import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import norm
from wrapping.plot_cylinder import *
from wrapping.algorithm import*
from wrapping.step_1 import find_cylinder_frame, find_matrix
from wrapping.Cylinder import Cylinder

####################
import biorbd
import bioviz

# Pseudo code : 
# On choisit le muscle que l’on veut 
# Ce muscle a un point d'origin et un point d’insertion, P et S, que l'on récupère
# Pour ce muscle, il existe 1 ou 2 cylindres avec lequel il va falloir faire un wrapping, on les nomme dans l’ordre U et V
# On va ensuite faire une boucle for pour récupérer ce dont on a besoin 
# Un q radom définit en fonction de ses bornes min et max
# Le P et S correspondant à ce q
# La longueur du chemin de passage
# On va mettre tout cela dans une matrice tel que on est (cf doc)

# essai : 
# segment_names
# ['thorax', 'spine', 'clavicle_effector_right', 'clavicle_right', 'scapula_effector_right', 'scapula_right', 'humerus_right', 'ulna_effector_right', 'ulna_right', 'radius_effector_right', 'radius_right', 'hand_right']

# Datas pour le cylindre (à priori) du thorax pour PECM2 et PECM3 (à partir de deux points)
C_T_PECM_1 = np.array([0.0187952455, -0.0992656744, 0.0784311931])
C_T_PECM_2 = np.array([0.0171630409, -0.014154527, 0.0749634019])

# Datas pour le cylindre (à piori) de l'humerus pour PECM1, PECM2 et PECM3 (à partir de deux points)
C_H_PECM_1 = np.array([-0.0452034817, -0.0711088305, 0.175903012])
C_H_PECM_2 = np.array([-0.0273048302, -0.0009948723, 0.1717388404])

cylinder_T_PECM = Cylinder(0.025, 1, C_T_PECM_1, C_T_PECM_2, "thorax")
cylinder_H_PECM = Cylinder(0.018, 1, C_H_PECM_1, C_H_PECM_2, "humerus_right")

# liste des cylindres pour PECM2 (et PECM3)
cylinders_PECM2=[cylinder_T_PECM, cylinder_H_PECM]

model = biorbd.Model("models/Wu_DeGroote.bioMod")

# Noms de tous les segments (= les os) du modèle
segment_names = [model.segment(i).name().to_string() for i in range(model.nbSegment())]
humerus_index = segment_names.index("humerus_right") # pour trouver où est humerus_index --» 6

humerus_dof_names = [model.segment(humerus_index).nameDof(i).to_string() for i in range(model.segment(humerus_index).nbQ())]
q_ranges = [[ranges.min(), ranges.max()] for ranges in model.segment(humerus_index).QRanges()]

# On choisit le muscle que l’on veut 
# Pour le moment, on va faire qu'avec pecm1 et pecm2 (ont leurs cylindres en commun)
def data_for_learning (muscle_selected, cylinders, num_q) :

   # On recupere la position du muscle
   muscle_names = [model.muscle(i).name().to_string() for i in range(model.nbMuscles())]
   muscle_index = muscle_names.index(muscle_selected) # on recupere le num/position du muscle dans le modele
   
   # On recupere la position des segments pour les cylindres U et V
   segment_names = [model.segment(i).name().to_string() for i in range(model.nbSegment())]
   segment_U_index = segment_names.index(cylinders[0].segment) 
   segment_V_index = segment_names.index(cylinders[1].segment) 
   
   # On recupere le ref initial des segments  
   q = np.zeros((model.nbQ(), ))
   gcs_seg_U_0 = [gcs.to_array() for gcs in model.allGlobalJCS(q)][segment_U_index]
   gcs_seg_V_0 = [gcs.to_array() for gcs in model.allGlobalJCS(q)][segment_V_index]
   
   for n in range (num_q) : 
      print("n = ", n, "ola")
      
      # faire un nouveau q random en respectant q_ranges
      # min_vals = [row[0] for row in q_ranges]
      # max_vals = [row[1] for row in q_ranges] 
      # q = np.random.uniform(low=min_vals, high=max_vals)
      
      # q[1]=-2
      
      print("q = ", q)
      
      # On met à jour les muscles par rapport à q
      model.updateMuscles(q) 
      mus = model.muscle(muscle_index) 
      
      # On met à jour les ref des segments pour le nouveau q
      model.UpdateKinematicsCustom(q) # pour mettre à jour par rapport au nouveau q
      gcs_seg_U = [gcs.to_array() for gcs in model.allGlobalJCS(q)][segment_U_index]
      gcs_seg_V = [gcs.to_array() for gcs in model.allGlobalJCS(q)][segment_V_index]
      # ------------------------------------------------
   
      # Ce muscle a un point d'origin et un point d’insertion, P et S, que l'on récupère
      # à chaque fois que l'on change de q, il faut remettre à jour, on recupere P et S
      origin_muscle = mus.musclesPointsInGlobal(model, q)[0].to_array() # le premier point est celui de l'origin
      insertion_muscle = mus.musclesPointsInGlobal(model, q)[-1].to_array() # le dernier est forcement insertion
      
      # en fonction de si on a 1 ou 2 cylindres, on va faire tel ou tel algo
      # on recupere cylindres au'on a besoin ( cylindre objet ?)
      # pour le moment on triche, on sait que pour PECM2 (ou PECM3) c'est cylinder_H_PECM (mobile) et cylinder_T_PECM (fixe)
      # cylinder_T_PECM (fixe) est tjs le meme peu importe q (bool == false)
      # cylinder_H_PECM (mobile) va bouger a chaque q, par defaut, est dans le ref de O
      
      # changer le ref du cylindre mobile en fonction du nouveau q (pas propre pour le moment ...)

      cylinder_U = np.dot(gcs_seg_U, np.dot(np.linalg.inv(gcs_seg_U_0), cylinders[0].matrix))
      cylinder_V = np.dot(gcs_seg_V, np.dot(np.linalg.inv(gcs_seg_V_0), cylinders[1].matrix))
      
      Q, G, H, T, Q_G_inactive, H_T_inactive, segment_length = double_cylinder_obstacle_set_algorithm(origin_muscle, insertion_muscle, cylinder_U,cylinders[0].radius, cylinders[0].side, cylinder_V, cylinders[1].radius, cylinders[1].side, np.dot(np.linalg.inv(cylinder_V), cylinder_U) )
      
      print("origin_point = ", origin_muscle)
      print("final_point = ", insertion_muscle)
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
      
      
      # on a alors les deux ref pour les cylindres (dans le ref de O !)
      # on utilise l'algo qui convient
      
      # on recupere la longueur du chemin de passage
      
      # fin de la boucle
   
   # return un tableau avec num_q lignes et avec les infos que l'on veut
   

   return None

data_for_learning ("PECM2",cylinders_PECM2, 1)

# Show
b = bioviz.Viz(loaded_model=model)
b.set_q(q)
b.exec()

exit(0)
#################

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

   matrix_UV = np.dot(np.linalg.inv(matrix_V), matrix_U)

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