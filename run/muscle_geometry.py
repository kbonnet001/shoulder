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

# Datas pour le cylindre (à piori) de l'humerus pour PECM1, PECM2 et PECM3 (à partir de deux points)
C_H_PECM2_1 = np.array([-0.0452034817, -0.0711088305, 0.175903012])
C_H_PECM2_2 = np.array([-0.0273048302, -0.0009948723, 0.1717388404])

cylinder_T_PECM2 = Cylinder(0.025, -1, C_T_PECM2_1, C_T_PECM2_2, "thorax")
cylinder_H_PECM2 = Cylinder(0.0175, 1, C_H_PECM2_1, C_H_PECM2_2, "humerus_right")

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
# -------------------

# On choisit le muscle que l’on veut 
# Pour le moment, on va faire qu'avec pecm2 
def data_for_learning (muscle_selected, cylinders, q, q_ranges_muscle, num_q, test = False) :

   # changement, rotation à cause de l'axe z --» axe y
   # l'algo ne peut fonctionner correctement sans ce changement
   matrix_rot_zy = np.array([[0,0,1,0], [1,0,0,0], [0,1,0,0], [0,0,0,1]])

   # On recupere la position du muscle
   muscle_names = [model.muscle(i).name().to_string() for i in range(model.nbMuscles())]
   muscle_index = muscle_names.index(muscle_selected) # on recupere le num/position du muscle dans le modele
   
   # On recupere la position des segments pour les cylindres U et V
   segment_names = [model.segment(i).name().to_string() for i in range(model.nbSegment())]
   
   if (len(cylinders) >= 1 and len(cylinders) <= 2) : 
      segment_U_index = segment_names.index(cylinders[0].segment) 
      gcs_seg_U_0 = [gcs.to_array() for gcs in model.allGlobalJCS(q)][segment_U_index] # ref initial des segments  
      
   if (len(cylinders) == 2) : 
      segment_V_index = segment_names.index(cylinders[1].segment) 
      gcs_seg_V_0 = [gcs.to_array() for gcs in model.allGlobalJCS(q)][segment_V_index] # ref initial des segments  
   
   # On prepare les limites de q
   min_vals = [row[0] for row in q_ranges_muscle]
   max_vals = [row[1] for row in q_ranges_muscle] 
   
   # Preparation des etiquettes pour le data frame
   qs = []
   Ps = []
   Ss = []
   Qs = []
   Gs = []
   Hs = []
   Ts = []
   Q_T_inactives = []
   Q_G_inactives = []
   H_T_inactives = []
   cylinder_Us = []
   cylinder_Vs = []
   segment_lengths = []
   
   for n in range (num_q) : 
         
      # faire un nouveau q random en respectant q_ranges
      q = np.random.uniform(low=min_vals, high=max_vals)
      
      # Tests -------------
      # pour test, faire une boucle avant à la place de range (num_q)
      #    for i in range (3) :
            # for j in range (3) :
            #    for k in range (3) :
      # q = np.array([q_test_limite[0][i],q_test_limite[1][j], q_test_limite[2][k], 0])
      # ------------------------
      
      # On met à jour les muscles par rapport à q
      model.updateMuscles(q) 
      mus = model.muscle(muscle_index) 
      
      # On met à jour les ref des segments pour le nouveau q
      model.UpdateKinematicsCustom(q) # pour mettre à jour par rapport au nouveau q

      # Ce muscle a un point d'origin et un point d’insertion, P et S, que l'on récupère
      origin_muscle = mus.musclesPointsInGlobal(model, q)[0].to_array() 
      insertion_muscle = mus.musclesPointsInGlobal(model, q)[-1].to_array() 
      # rotation
      origin_muscle_rot = np.dot(matrix_rot_zy[0:3, 0:3], origin_muscle)
      insertion_muscle_rot = np.dot(matrix_rot_zy[0:3, 0:3], insertion_muscle)
      
      # ------------------------------------------------

      if (len(cylinders) == 0) :
         # le chemin de passage est juste une ligne droite
         segment_length = norm(np.array(origin_muscle_rot) - np.array(insertion_muscle_rot))
   
      elif (len(cylinders) == 1) :
         gcs_seg_U = [gcs.to_array() for gcs in model.allGlobalJCS(q)][segment_U_index]
         
         # Changement de ref des cylindres en fonction de leur segment
         cylinder_U = np.dot(gcs_seg_U, np.dot(np.linalg.inv(gcs_seg_U_0), cylinders[0].matrix))
         cylinder_U_rot = np.dot(matrix_rot_zy, cylinder_U)
         
         Q_rot, T_rot, Q_T_inactive, segment_length = single_cylinder_obstacle_set_algorithm(origin_muscle_rot, insertion_muscle_rot, cylinders[0].radius, cylinders[0].side, cylinder_U_rot)
      
         Qs.append(np.dot(np.transpose(matrix_rot_zy[0:3, 0:3]), Q_rot))
         Ts.append(np.dot(np.transpose(matrix_rot_zy[0:3, 0:3]), T_rot))
         Q_T_inactives.append(Q_T_inactive)
         cylinder_Us.append(cylinder_U)
      
         if test == True : 
            plot_one_cylinder_obstacle(origin_muscle_rot, insertion_muscle_rot, cylinders[0].radius, Q_rot, T_rot, Q_T_inactive, cylinder_U_rot)
         
      else : # (len(cylinders) == 2 ) 
         gcs_seg_U = [gcs.to_array() for gcs in model.allGlobalJCS(q)][segment_U_index]
         gcs_seg_V = [gcs.to_array() for gcs in model.allGlobalJCS(q)][segment_V_index]
      
         # Changement de ref des cylindres en fonction de leur segment
         cylinder_U = np.dot(gcs_seg_U, np.dot(np.linalg.inv(gcs_seg_U_0), cylinders[0].matrix))
         cylinder_V = np.dot(gcs_seg_V, np.dot(np.linalg.inv(gcs_seg_V_0), cylinders[1].matrix))
         
         cylinder_U_rot = np.dot(matrix_rot_zy, cylinder_U)
         cylinder_V_rot = np.dot(matrix_rot_zy, cylinder_V)
         
         Q_rot, G_rot, H_rot, T_rot, Q_G_inactive, H_T_inactive , segment_length  = double_cylinder_obstacle_set_algorithm(origin_muscle_rot, insertion_muscle_rot, cylinder_U_rot,cylinders[0].radius, cylinders[0].side, cylinder_V_rot, cylinders[1].radius, cylinders[1].side, np.dot(np.linalg.inv(cylinder_V_rot), cylinder_U_rot) )
      
         Qs.append(np.dot(np.transpose(matrix_rot_zy[0:3, 0:3]), Q_rot))
         Gs.append(np.dot(np.transpose(matrix_rot_zy[0:3, 0:3]), G_rot))
         Hs.append(np.dot(np.transpose(matrix_rot_zy[0:3, 0:3]), H_rot))
         Ts.append(np.dot(np.transpose(matrix_rot_zy[0:3, 0:3]), T_rot))
         Q_G_inactives.append(Q_G_inactive)
         H_T_inactives.append(H_T_inactive)
         cylinder_Us.append(cylinder_U)
         cylinder_Vs.append(cylinder_V)

         if test == True : 
            plot_double_cylinder_obstacle(origin_muscle_rot, insertion_muscle_rot, cylinders[0].radius, cylinders[1].radius, Q_rot, G_rot, H_rot, T_rot, cylinder_U_rot, cylinder_V_rot, Q_G_inactive, H_T_inactive)
      
      qs.append(q)
      Ps.append(np.dot(np.transpose(matrix_rot_zy[0:3, 0:3]), origin_muscle_rot))
      Ss.append(np.dot(np.transpose(matrix_rot_zy[0:3, 0:3]), insertion_muscle_rot))
      segment_lengths.append(segment_length)
      
      # print("origin_point = ", origin_muscle)
      # print("final_point = ", insertion_muscle)
      # print("Q = ", Q)
      # print("G = ", G)
      # print("H = ", H)
      # print("T = ", T)
      # print("Q_G_inactive = ",Q_G_inactive)
      # print("H_T_inactive = ",H_T_inactive)
      # if (Q_G_inactive and H_T_inactive) :
      #    print("--> Straight line")
      # elif (Q_G_inactive) :
      #    print("--> Single cylinder algo with V")
      # elif (H_T_inactive) :
      #    print("--> Single cylinder algo with U")
      # else :
      #    print("--> Double cylinder algo with U and V")
      #    print("segment_length = ", round(segment_length, 2))
            
   
   # return un tableau avec num_q lignes et avec les infos que l'on veut
   if (len(cylinders ) == 0) :
      data = {
         "q": qs,
         "P": Ps,
         "S": Ss,
         "segment_length": segment_lengths
      }
      
   elif (len(cylinders) == 1) :
      data = {
         "q": qs,
         "P": Ps,
         "S": Ss,
         "Q": Qs,
         "T": Ts,
         "Q_T_inactive": Q_G_inactives,
         "cylinder_Us": cylinder_Us, 
         "segment_length": segment_lengths
      }
   else :
      data = {
         "q": qs,
         "P": Ps,
         "S": Ss,
         "Q": Qs,
         "G": Gs,
         "H": Hs,
         "T": Ts,
         "Q_G_inactive": Q_G_inactives,
         "H_T_inactive": H_T_inactives,
         "cylinder_Us": cylinder_Us, 
         "cylinder_Vs": cylinder_Vs,
         "segment_length": segment_lengths
      }
         
   #load data into a DataFrame object: 
   df_test_limites_q = pd.DataFrame(data)

   print(df_test_limites_q) 
   if test == False : 
      df_test_limites_q.to_excel(f"df_{muscle_selected}_datas.xlsx")

   return None

data_for_learning ("PECM2",cylinders_PECM2, q, q_ranges_PECM2, 10)

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