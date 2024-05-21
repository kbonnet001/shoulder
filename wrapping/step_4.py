import numpy as np
from scipy.linalg import norm
from wrapping.step_1 import switch_frame
from wrapping.step_2 import compute_length_v1_v2_xy

# Functions for Step 4
#----------------------

def compute_length_v1_v2(v1,v2, v1_v2_length_xy) :

  # Compute length of path segments v1 v2
  #
  # INPUT
  # - v1 : array 3*1 position of the first obstacle tangent point
  # - v2 : array 3*1 position of the second obstacle tangent point
  # - v1_v2_length_xy : xy coordinates of segment lengths in plane
  #
  # OUTPUT
  # - ||v1v2|| : length of path segments between v1 and v2

  return np.sqrt(v1_v2_length_xy**2+(v2[0]-v1[2])**2)

def segment_length_single_cylinder(obstacle_tangent_point_inactive, origin_point, final_point, v1, v2, r) :

  # Compute length of path segments
  #
  # INPUT
  # - obstacle_tangent_point_inactive : bool determine if v1 and v1 or inactive (True) or not (False)
  # - origin_point : array 3*1 position of the first point
  # - final_point : array 3*1 position of the second point
  # - v1 : array 3*1 position of the first obstacle tangent point
  # - v2 : array 3*1 position of the second obstacle tangent point
  # - r : radius of the cylinder * side
  #
  # OUTPUT
  # - segment_length : length of path segments

  if (obstacle_tangent_point_inactive == True) : # Muscle path is straight line from origin_point to final_point
   segment_length = norm(np.array(final_point)-np.array(origin_point))

  else :
   v1_v2_length_xy = compute_length_v1_v2_xy(v1, v2, r)
   v1_v2_length = compute_length_v1_v2(v1,v2, v1_v2_length_xy)
   segment_length = norm (v1 - np.array(origin_point)) + v1_v2_length + norm(np.array(final_point) - v2)

  return segment_length

def segment_length_double_cylinder(Q_G_inactive, H_T_inactive, P, S, P_U_cylinder_frame, P_V_cylinder_frame, S_U_cylinder_frame, S_V_cylinder_frame, Q, G, H, T, r_U, r_V, matrix_U, matrix_V) :

   # Compute length of path segments
   #
   # INPUT
   # - Q_G_inactive : bool determine if Q and G or inactive (True) or not (False)
   # - H_T_inactive : bool determine if H and T or inactive (True) or not (False)
   # - P : array 3*1 position of the first point
   # - S : array 3*1 position of the second point
   # - P_U_cylinder_frame : array 3*1 position of the first point in U cylinder frame
   # - P_V_cylinder_frame : array 3*1 position of the first point in V cylinder frame
   # - S_U_cylinder_frame : array 3*1 position of the second point in U cylinder frame
   # - S_V_cylinder_frame : array 3*1 position of the second point in V cylinder frame
   # - Q : array 3*1 position of the first obstacle tangent point (in V cylinder frame)
   # - G : array 3*1 position of the second obstacle tangent point (in V cylinder frame)
   # - H : array 3*1 position of the third obstacle tangent point (in U cylinder frame)
   # - T : array 3*1 position of the fourth obstacle tangent point (in U cylinder frame)
   # - r_V : radius of the cylinder U * side_U
   # - r_U : radius of the cylinder V * side_V
   # - matrix_U : array 4*4 rotation_matrix and vect for cylinder U
   # - matrix_V : array 4*4 rotation_matrix and vect for cylinder V
   #
   # OUTPUT
   # - segment_length : length of path segments

   # Compute lengths
   H_T_length_xy = compute_length_v1_v2_xy(H, T, r_V)
   H_T_length = compute_length_v1_v2(H, T, H_T_length_xy)
   Q_G_length_xy = compute_length_v1_v2_xy(Q, G, r_U)
   Q_G_length = compute_length_v1_v2(Q, G, Q_G_length_xy)

   # Compute segment lengths based on conditions
   if Q_G_inactive and H_T_inactive: # Muscle path is straight line from origin_point to final_point
       segment_length = norm(np.array(P) - np.array(S))

   elif Q_G_inactive: # single cylinder algorithm with V cylinder
       segment_length = norm(H - np.array(P_V_cylinder_frame)) + H_T_length + norm(np.array(S_V_cylinder_frame) - T)

   elif H_T_inactive: # single cylinder algorithm with U cylinder
       segment_length = norm(Q - np.array(P_U_cylinder_frame)) + Q_G_length + norm(np.array(S_U_cylinder_frame) - G)

   else: # double cylinder
       H_T_length_xy = compute_length_v1_v2_xy(H, T, r_V)
       H_T_length = compute_length_v1_v2(H, T, H_T_length_xy)

       G_H_length = norm(switch_frame(H, matrix_V) - switch_frame(G, matrix_U))

       segment_length = norm(Q - np.array(P_U_cylinder_frame)) + Q_G_length + G_H_length + H_T_length_xy + norm(np.array(S_V_cylinder_frame) - T)

   return segment_length