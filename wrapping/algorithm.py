import numpy as np
from wrapping.step_1 import switch_frame, transpose_switch_frame
from wrapping.step_2 import find_tangent_points, compute_length_v1_v2_xy, find_tangent_points_iterative_method, point_inside_cylinder
from wrapping.step_3 import determine_if_tangent_points_inactive_single_cylinder
from wrapping.step_4 import segment_length_single_cylinder, segment_length_double_cylinder


# Algorithm
#---------------------------

def single_cylinder_obstacle_set_algorithm(origin_point, final_point, radius, side, cylinder_origin, cylinder_frame) :

   # Provide the length wrapping around a cylinder
   #  Based on:
   #  -B.A. Garner and M.G. Pandy, The obstacle-set method for
   #  representing muscle paths in musculoskeletal models,
   #  Comput. Methods Biomech. Biomed. Engin. 3 (2000), pp. 1–30.
   # -----------------------------------------------------------
   #
   # INPUT
   # - origin_point : array 3*1 position of the first point
   # - final_point : array 3*1 position of the second point
   # - radius : radius of the cylinder
   # - side : side of the wrapping, -1 for the left side, 1 for the right side
   # - cylinder_origin : array 3*1 coordinates of the center of the cylinder
   # - cylinder_frame : array 3*3 local frame of the cylinder
   #
   # OUTPUT
   # - v1o : array 3*1 position of the first obstacle tangent point (in conventionnal frame)
   # - v2o : array 3*1 position of the second obstacle tangent point (in conventionnal frame)
   # - obstacle_tangent_point_inactive : bool determine if v1 and v1 or inactive (True) or not (False)
   # - segment_lenght : length of path segments

   # ------
   # Step 1
   # ------
   r = radius * side

   # Express P and S in the cylinder frame
   P_cylinder_frame = transpose_switch_frame(origin_point, cylinder_frame, [0,0,0] - cylinder_origin)
   S_cylinder_frame = transpose_switch_frame(final_point, cylinder_frame, [0,0,0] - cylinder_origin)

   # ------
   # Step 2
   # ------
   # tangent points
   v1, v2 = find_tangent_points(P_cylinder_frame, S_cylinder_frame, r)
   v1_v2_length_xy = compute_length_v1_v2_xy(v1, v2, r)

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
   v1o = switch_frame(v1, np.transpose(cylinder_frame), cylinder_origin)
   v2o = switch_frame(v2, np.transpose(cylinder_frame), cylinder_origin)

   return v1o, v2o, obstacle_tangent_point_inactive, segment_length

def double_cylinder_obstacle_set_algorithm(P, S, U_origin, cylinder_frame_U, radius_U, side_U, V_origin, cylinder_frame_V, radius_V, side_V, rotation_matrix_UV) :

   # Provide the length wrapping around a cylinder
   #  Based on:
   #  -B.A. Garner and M.G. Pandy, The obstacle-set method for
   #  representing muscle paths in musculoskeletal models,
   #  Comput. Methods Biomech. Biomed. Engin. 3 (2000), pp. 1–30.
   # -----------------------------------------------------------
   #
   # INPUT
   # - P : array 3*1 position of the first point
   # - S : array 3*1 position of the second point
   # - U_origin : array 3*1 coordinates of the center of the cylinder U
   # - cylinder_frame_U : array 3*3 ortonormal frame for the cylinder U
   # - radius_U : radius of the cylinder U
   # - side_U : side of the wrapping (cylinder U), -1 for the left side, 1 for the right side
   # - V_origin : array 3*1 coordinates of the center of the cylinder V
   # - cylinder_frame_V : array 3*3 ortonormal frame for the cylinder V
   # - radius_V : radius of the cylinder V
   # - side_V : side of the wrapping (cylinder V), -1 for the left side, 1 for the right side
   # - rotation_matrix_UV : array 3*3 rotation matrix to change frame (U --> V)
   #
   # OUTPUT
   # - Qo : array 3*1 position of the first obstacle tangent point (in conventional frame)
   # - Go : array 3*1 position of the second obstacle tangent point (in conventional frame)
   # - Ho : array 3*1 position of the third obstacle tangent point (in conventional frame)
   # - To : array 3*1 position of the fourth obstacle tangent point (in conventional frame)
   # - Q_G_inactive : bool determine if Q and G or inactive (True) or not (False)
   # - H_T_inactive : bool determine if H and T or inactive (True) or not (False)
   # - segment_lenght : length of path segments

   # ------
   # Step 1
   # ------
   r_U = radius_U * side_U
   r_V = radius_V * side_V
   origin_V_in_U_frame = transpose_switch_frame(V_origin, cylinder_frame_U, [0,0,0] - U_origin)
   origin_U_in_V_frame = transpose_switch_frame(U_origin, cylinder_frame_V, [0,0,0] - V_origin)

   # Express P (S) in U (V) cylinder frame
   P_U_cylinder_frame = transpose_switch_frame(P, cylinder_frame_U, [0,0,0] - U_origin)
   P_V_cylinder_frame = transpose_switch_frame(P, cylinder_frame_V, [0,0,0] - V_origin)

   S_U_cylinder_frame = transpose_switch_frame(S, cylinder_frame_U, [0,0,0] - U_origin)
   S_V_cylinder_frame = transpose_switch_frame(S, cylinder_frame_V, [0,0,0] - V_origin)

   # ------
   # Step 2
   # ------

   point_inside_U = point_inside_cylinder(P_U_cylinder_frame, S_U_cylinder_frame, radius_U)
   point_inside_V = point_inside_cylinder(P_V_cylinder_frame, S_V_cylinder_frame, radius_V)

   if point_inside_U and point_inside_V :
    print("You choose P and/or S in the cylinder U and V. Muscle path is straight line")
    Q, G, H, T = [0,0,0], [0,0,0], [0,0,0], [0,0,0]

   elif point_inside_U :
    print("You choose P and/or S in the cylinder U. Muscle path is straight line")
    Q, G = [0,0,0], [0,0,0]
    H, T = find_tangent_points(P_V_cylinder_frame, S_V_cylinder_frame, r_V)

   elif point_inside_V :
    print("You choose P and/or S in the cylinder V. Muscle path is straight line")
    H, T = [0,0,0], [0,0,0]
    Q, G = find_tangent_points(P_U_cylinder_frame, S_U_cylinder_frame, r_U)

   else :
    Q, G, H, T = find_tangent_points_iterative_method(P_U_cylinder_frame, S_V_cylinder_frame, r_V, r_U, rotation_matrix_UV, origin_V_in_U_frame, origin_U_in_V_frame )

   # ------
   # Step 3
   # ------
   Q_G_inactive = determine_if_tangent_points_inactive_single_cylinder(Q, G, r_U)
   H_T_inactive = determine_if_tangent_points_inactive_single_cylinder(H, T, r_V)

  #  if Q_G_inactive==True :
  #   H, T = find_tangent_points(P_V_cylinder_frame, S_V_cylinder_frame, r_V)
  #   H_T_inactive = determine_if_tangent_points_inactive_single_cylinder(H, T, r_V)

  #  if H_T_inactive==True :
  #    Q, G = find_tangent_points(P_U_cylinder_frame, S_U_cylinder_frame, r_U)
  #    Q_G_inactive = determine_if_tangent_points_inactive_single_cylinder(Q, G, r_U)

   # ------
   # Step 4
   # ------
   segment_length = segment_length_double_cylinder(Q_G_inactive, H_T_inactive, P, S, P_U_cylinder_frame, P_V_cylinder_frame, S_U_cylinder_frame, S_V_cylinder_frame, Q, G, H, T, r_U, r_V, cylinder_frame_U, cylinder_frame_V, U_origin, V_origin)

   # ------
   # Step 5
   # ------
   Qo = switch_frame(Q, np.transpose(cylinder_frame_U), U_origin)
   Go = switch_frame(G, np.transpose(cylinder_frame_U), U_origin)
   Ho = switch_frame(H, np.transpose(cylinder_frame_V), V_origin)
   To = switch_frame(T, np.transpose(cylinder_frame_V), V_origin)

   return Qo, Go, Ho, To, Q_G_inactive, H_T_inactive, segment_length