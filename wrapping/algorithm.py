from wrapping.step_1 import switch_frame, transpose_switch_frame
from wrapping.step_2 import find_tangent_points, compute_length_v1_v2_xy, find_tangent_points_iterative_method, point_inside_cylinder, find_tangent_points_iterative_method_2
from wrapping.step_3 import determine_if_tangent_points_inactive_single_cylinder
from wrapping.step_4 import segment_length_single_cylinder, segment_length_double_cylinder


# Algorithm
#---------------------------

def single_cylinder_obstacle_set_algorithm(origin_point, final_point, radius, side, matrix) :

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
   r = radius * side

   # Express P and S in the cylinder frame
   P_cylinder_frame = transpose_switch_frame(origin_point, matrix)
   S_cylinder_frame = transpose_switch_frame(final_point, matrix)

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
   v1o = switch_frame(v1, matrix)
   v2o = switch_frame(v2, matrix)

   return v1o, v2o, obstacle_tangent_point_inactive, segment_length

def double_cylinder_obstacle_set_algorithm(P, S, matrix_U, radius_U, side_U, matrix_V, radius_V, side_V, rotation_matrix_UV) :

   """Provide the length wrapping around a cylinder
    Based on:
    -B.A. Garner and M.G. Pandy, The obstacle-set method for
    representing muscle paths in musculoskeletal models,
    Comput. Methods Biomech. Biomed. Engin. 3 (2000), pp. 1–30.
   -----------------------------------------------------------
   
   INPUT
   - P : array 3*1 position of the first point
   - S : array 3*1 position of the second point
   - matrix_U : array 4*4 rotation_matrix and vect for cylinder U
   - radius_U : radius of the cylinder U
   - side_U : side of the wrapping (cylinder U), -1 for the left side, 1 for the right side
   - matrix_V : array 4*4 rotation_matrix and vect for cylinder V
   - radius_V : radius of the cylinder V
   - side_V : side of the wrapping (cylinder V), -1 for the left side, 1 for the right side
   - rotation_matrix_UV : array 3*3 rotation matrix to change frame (U --> V)
   
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
   r_U = radius_U * side_U
   r_V = radius_V * side_V

   # Express P (S) in U (V) cylinder frame
   P_U_cylinder_frame = transpose_switch_frame(P, matrix_U)
   P_V_cylinder_frame = transpose_switch_frame(P, matrix_V)

   S_U_cylinder_frame = transpose_switch_frame(S, matrix_U)
   S_V_cylinder_frame = transpose_switch_frame(S, matrix_V)

   # ------
   # Step 2
   # ------

   point_inside_U = point_inside_cylinder(P_U_cylinder_frame, S_U_cylinder_frame, radius_U)
   point_inside_V = point_inside_cylinder(P_V_cylinder_frame, S_V_cylinder_frame, radius_V)

   if point_inside_U and point_inside_V :
    print("You choose P and/or S in the cylinder U and V. Muscle path is straight line")
    Q, G, H, T = [0,0,0], [0,0,0], [0,0,0], [0,0,0]

   elif point_inside_U :
    print("You choose P in the cylinder U. Muscle path is straight line")
    Q, G = [0,0,0], [0,0,0]
    H, T = find_tangent_points(P_V_cylinder_frame, S_V_cylinder_frame, r_V)

   elif point_inside_V :
    print("You choose S in the cylinder V. Muscle path is straight line")
    H, T = [0,0,0], [0,0,0]
    Q, G = find_tangent_points(P_U_cylinder_frame, S_U_cylinder_frame, r_U)

   else :
    Q, G, H, T = find_tangent_points_iterative_method_2(P_U_cylinder_frame,P_V_cylinder_frame, S_U_cylinder_frame, S_V_cylinder_frame, r_V, r_U, rotation_matrix_UV, matrix_U, matrix_V)

   # ------
   # Step 3
   # ------
   Q_G_inactive = determine_if_tangent_points_inactive_single_cylinder(Q, G, r_U)
   H_T_inactive = determine_if_tangent_points_inactive_single_cylinder(H, T, r_V)

   if Q_G_inactive==True :
    H, T = find_tangent_points(P_V_cylinder_frame, S_V_cylinder_frame, r_V)
    H_T_inactive = determine_if_tangent_points_inactive_single_cylinder(H, T, r_V)

   if H_T_inactive==True :
     Q, G = find_tangent_points(P_U_cylinder_frame, S_U_cylinder_frame, r_U)
     Q_G_inactive = determine_if_tangent_points_inactive_single_cylinder(Q, G, r_U)

   # ------
   # Step 4
   # ------
   segment_length = segment_length_double_cylinder(Q_G_inactive, H_T_inactive, P, S, P_U_cylinder_frame, P_V_cylinder_frame, S_U_cylinder_frame, S_V_cylinder_frame, Q, G, H, T, r_U, r_V, matrix_U, matrix_V)

   # ------
   # Step 5
   # ------
   Qo = switch_frame(Q, matrix_U)
   Go = switch_frame(G, matrix_U)
   Ho = switch_frame(H, matrix_V)
   To = switch_frame(T, matrix_V)

   return Qo, Go, Ho, To, Q_G_inactive, H_T_inactive, segment_length