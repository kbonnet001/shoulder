from wrapping.step_1 import switch_frame, transpose_switch_frame
from wrapping.step_2 import *
from wrapping.step_3 import determine_if_tangent_points_inactive_single_cylinder
# from wrapping.step_4 import segment_length_single_cylinder
from wrapping.plot_cylinder import plot_double_cylinder_obstacle
import numpy as np
from wrapping.paspropre import determine_if_needed_change_side, determine_if_needed_change_side_2


# Algorithm
#---------------------------

def single_cylinder_obstacle_set_algorithm(origin_point, final_point, Cylinder) :

   """Provide the length wrapping around a cylinder
    Based on:
    -B.A. Garner and M.G. Pandy, The obstacle-set method for
    representing muscle paths in musculoskeletal models,
    Comput. Methods Biomech. Biomed. Engin. 3 (2000), pp. 1–30.
   -----------------------------------------------------------
   
   INPUT
   - origin_point : array 3*1 position of the first point
   - final_point : array 3*1 position of the second point
   - radius : radius of the cylinder
   - side : side of the wrapping, -1 for the left side, 1 for the right side
   - matrix : array 4*4 rotation_matrix and vect
   
   OUTPUT
   - v1o : array 3*1 position of the first obstacle tangent point (in conventionnal frame)
   - v2o : array 3*1 position of the second obstacle tangent point (in conventionnal frame)
   - obstacle_tangent_point_inactive : bool determine if v1 and v1 or inactive (True) or not (False)
   - segment_lenght : length of path segments"""

   # ------
   # Step 1
   # ------
   r = Cylinder.radius * Cylinder.side

   # Express P and S in the cylinder frame
   P_cylinder_frame = transpose_switch_frame(origin_point, Cylinder.matrix)
   S_cylinder_frame = transpose_switch_frame(final_point, Cylinder.matrix)

   # ------
   # Step 2
   # ------
   # tangent points
   v1, v2 = find_tangent_points(P_cylinder_frame, S_cylinder_frame, r)

   # ------
   # Step 3
   # ------
   obstacle_tangent_point_inactive = determine_if_tangent_points_inactive_single_cylinder(v1,v2, r)

   # ------
   # Step 4
   # ------
   segment_length = segment_length_single_cylinder(obstacle_tangent_point_inactive, P_cylinder_frame, S_cylinder_frame, v1, v2, r)

   # ------
   # Step 5
   # ------
   v1o = switch_frame(v1, Cylinder.matrix)
   v2o = switch_frame(v2, Cylinder.matrix)

   return v1o, v2o, obstacle_tangent_point_inactive, segment_length

def double_cylinder_obstacle_set_algorithm(P, S, Cylinder_U, Cylinder_V, list_ref = []) :

   """Provide the length wrapping around a cylinder
    Based on:
    -B.A. Garner and M.G. Pandy, The obstacle-set method for
    representing muscle paths in musculoskeletal models,
    Comput. Methods Biomech. Biomed. Engin. 3 (2000), pp. 1–30.
   -----------------------------------------------------------
   
   INPUT
   - P : array 3*1 position of the first point
   - S : array 3*1 position of the second point
   - Cylinder_U.matrix : array 4*4 rotation_matrix and vect for cylinder U
   - Cylinder_U.radius : radius of the cylinder U
   - side_U : side of the wrapping (cylinder U), -1 for the left side, 1 for the right side
   - Cylinder_V.matrix : array 4*4 rotation_matrix and vect for cylinder V
   - Cylinder_V.radius : radius of the cylinder V
   - side_V : side of the wrapping (cylinder V), -1 for the left side, 1 for the right side
   - rotation_Cylinder_U.matrixV : array 3*3 rotation matrix to change frame (U --> V)
   
   OUTPUT
   - Qo : array 3*1 position of the first obstacle tangent point (in conventional frame)
   - Go : array 3*1 position of the second obstacle tangent point (in conventional frame)
   - Ho : array 3*1 position of the third obstacle tangent point (in conventional frame)
   - To : array 3*1 position of the fourth obstacle tangent point (in conventional frame)
   - Q_G_inactive : bool determine if Q and G or inactive (True) or not (False)
   - H_T_inactive : bool determine if H and T or inactive (True) or not (False)
   - segment_lenght : length of path segments"""

   # ------
   # Step 1
   # ------
   r_U = Cylinder_U.radius * Cylinder_U.side
   r_V = Cylinder_V.radius * Cylinder_V.side

   # Express P (S) in U (V) cylinder frame
   P_U_cylinder_frame = transpose_switch_frame(P, Cylinder_U.matrix)
   P_V_cylinder_frame = transpose_switch_frame(P, Cylinder_V.matrix)

   S_U_cylinder_frame = transpose_switch_frame(S, Cylinder_U.matrix)
   S_V_cylinder_frame = transpose_switch_frame(S, Cylinder_V.matrix)
   # Cylinder_V.change_raidus(S_V_cylinder_frame[0])
   print("S_V_cylinder_frame = ", S_V_cylinder_frame)
   
   error_wrapping = False

   # ------
   # Step 2
   # ------

   epsilon = 0.00095
   #0.0008 #0.00095
   
   # P_inside_U = point_inside_cylinder(P_U_cylinder_frame, Cylinder_U.radius, epsilon)
   # S_inside_V = point_inside_cylinder(S_V_cylinder_frame, Cylinder_V.radius, epsilon)
   
   # if P_inside_U and S_inside_V :
   #    print("You choose P and/or S in the cylinder U and V. Muscle path is straight line")
   #    Q, G, H, T = [0,0,0], [0,0,0], [0,0,0], [0,0,0]

   # elif P_inside_U :
   #    print("You choose P in the cylinder U. Muscle path is straight line")
   #    Q, G = [0,0,0], [0,0,0]
   #    H, T = find_tangent_points(P_V_cylinder_frame, S_V_cylinder_frame, r_V)

   # elif S_inside_V:
   #    print("You choose S in the cylinder V. Muscle path is straight line")
   #    H, T = [0,0,0], [0,0,0]
   #    Q, G = find_tangent_points(P_U_cylinder_frame, S_U_cylinder_frame, r_U)

   # else :
   #    Q, G, H, T = find_tangent_points_iterative_method(P, S, P_U_cylinder_frame,P_V_cylinder_frame, S_U_cylinder_frame, S_V_cylinder_frame, r_V, r_U,  Cylinder_U.matrix, Cylinder_V.matrix)
   
   Q, G, H, T = find_tangent_points_iterative_method(P, S, P_U_cylinder_frame,P_V_cylinder_frame, S_U_cylinder_frame, S_V_cylinder_frame, r_V, r_U,  Cylinder_U.matrix, Cylinder_V.matrix)
   
   # ici, Q, G sont dans le local du cylindre U
   # G et H sont dans le repere local du cylindre V
   
   H_T_inactive = determine_if_tangent_points_inactive_single_cylinder(H, T, r_V)
   print("HT = ", H_T_inactive)
   
   # if H_T_inactive and determine_if_needed_change_side(S_V_cylinder_frame, np.array([0.0246330366, -0.0069265376, -0.0000168612])): 
   if H_T_inactive and determine_if_needed_change_side(S_V_cylinder_frame, np.array([0.0179188682, -0.0181428819, 0.02])) == True: 
      print("yop")
      Cylinder_V.change_side()
      r_V = Cylinder_V.radius * Cylinder_V.side
      Q, G, H, T = find_tangent_points_iterative_method(P, S, P_U_cylinder_frame,P_V_cylinder_frame, S_U_cylinder_frame, S_V_cylinder_frame, r_V, r_U,  Cylinder_U.matrix, Cylinder_V.matrix)
      Cylinder_V.change_side()
      # r_V = Cylinder_V.radius * Cylinder_V.side
      
   # elif H_T_inactive == False and determine_if_needed_change_side_2(S_V_cylinder_frame, np.array([0.0179188682, -0.0181428819, 0.02])) == True: 
   #    print("yop")
   #    Cylinder_V.change_side()
   #    r_V = Cylinder_V.radius * Cylinder_V.side
   #    Q, G, H, T = find_tangent_points_iterative_method(P, S, P_U_cylinder_frame,P_V_cylinder_frame, S_U_cylinder_frame, S_V_cylinder_frame, r_V, r_U,  Cylinder_U.matrix, Cylinder_V.matrix)
   #    Cylinder_V.change_side()
      
   # if H_T_inactive == False and determine_if_needed_change_side(S_V_cylinder_frame, np.array([0.0246330366, -0.0069265376, -0.0000168612])) == True: 
   #    Cylinder_V.change_side()
   #    Q, G, H, T = find_tangent_points_iterative_method(P, S, P_U_cylinder_frame,P_V_cylinder_frame, S_U_cylinder_frame, S_V_cylinder_frame, r_V, r_U,  Cylinder_U.matrix, Cylinder_V.matrix)
   #    Cylinder_V.change_side()
   
   
   
   # if list_ref != [] :
   #    # Faut mettre Tref dans le local de V
   #    # T_ref_local = transpose_switch_frame(list_ref[-1], Cylinder_V.matrix)
   #    # # T_ref_local = np.dot(np.transpose(Cylinder_V.matrix), list_ref[-1])
   #    # print("T = ", T)
   #    # # print("T ref = ", T_ref_local)
   #    # print("T ref = ", T_ref_local)
   #    print("ola")
      
   # utiliser list_ref
   # verifier si ok
   # si pas ok changer side
   # refaire find tangent points ...
   
   # print("H = ", H) #tout  semble innutile
   # if H == [0,0,0] : 
   #    print("ok")
   #    Cylinder_V.change_side()
   #    Q, G, H, T = find_tangent_points_iterative_method(P, S, P_U_cylinder_frame,P_V_cylinder_frame, S_U_cylinder_frame, S_V_cylinder_frame, r_V, r_U,  Cylinder_U.matrix, Cylinder_V.matrix)
   #    Cylinder_V.change_side()
      
   
   # ------
   # Step 3
   # ------
   Q_G_inactive = determine_if_tangent_points_inactive_single_cylinder(Q, G, r_U)
   H_T_inactive = determine_if_tangent_points_inactive_single_cylinder(H, T, r_V)

   # if Q_G_inactive==True :
   #    H, T = find_tangent_points(P_V_cylinder_frame, S_V_cylinder_frame, r_V)
   #    H_T_inactive = determine_if_tangent_points_inactive_single_cylinder(H, T, r_V)

   # if H_T_inactive==True :
   #    Q, G = find_tangent_points(P_U_cylinder_frame, S_U_cylinder_frame, r_U)
   #    Q_G_inactive = determine_if_tangent_points_inactive_single_cylinder(Q, G, r_U)
   #    if np.linalg.norm(G[:2]) < Cylinder_V.radius : 
   #       error_wrapping = True
   #       print("error = ", error_wrapping)

   if H_T_inactive==True :
      Q, G = find_tangent_points(P_U_cylinder_frame, S_U_cylinder_frame, r_U)
      Q_G_inactive = determine_if_tangent_points_inactive_single_cylinder(Q, G, r_U)
      # if point_tangent_inside_cylinder(G, Cylinder_U, Cylinder_V, epsilon=0.0) : 
      #    error_wrapping = True
      #    print("error = ", error_wrapping)
         
   elif Q_G_inactive==True :
      H, T = find_tangent_points(P_V_cylinder_frame, S_V_cylinder_frame, r_V)
      H_T_inactive = determine_if_tangent_points_inactive_single_cylinder(H, T, r_V)

   # ------
   # Step 4
   # ------
   
   segment_length = segment_length_double_cylinder(Q_G_inactive, H_T_inactive, P, S, P_U_cylinder_frame, P_V_cylinder_frame, S_U_cylinder_frame, S_V_cylinder_frame, Q, G, H, T, r_U, r_V, Cylinder_U.matrix, Cylinder_V.matrix)
   # segment_length = segment_length_double_cylinder2(Q_G_inactive, H_T_inactive, P, S, P_U_cylinder_frame, P_V_cylinder_frame, S_U_cylinder_frame, S_V_cylinder_frame, Q, G, H, T, r_U, r_V, Cylinder_U.matrix, Cylinder_V.matrix)

   # ------
   # Step 5
   # ------
   Qo = switch_frame(Q, Cylinder_U.matrix)
   Go = switch_frame(G, Cylinder_U.matrix)
   Ho = switch_frame(H, Cylinder_V.matrix)
   To = switch_frame(T, Cylinder_V.matrix)
   
   return Qo, Go, Ho, To, Q_G_inactive, H_T_inactive, segment_length

def angle_between_points(point1, point2):
    # Convertir les points en vecteurs à partir de l'origine
    vec1 = np.array(point1[:2])
    vec2 = np.array(point2[:2])
    
    # Calculer le produit scalaire des deux vecteurs
    dot_product = np.dot(vec1, vec2)
    
    # Calculer les normes des vecteurs
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    
    # Calculer le cosinus de l'angle entre les vecteurs
    cos_angle = dot_product / (norm_vec1 * norm_vec2)
    
    # Calculer l'angle en radians
    angle_radians = np.arccos(cos_angle)
    
    # Convertir l'angle en degrés
    angle_degrees = np.degrees(angle_radians)
    
    return angle_degrees