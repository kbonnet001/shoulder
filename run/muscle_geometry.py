import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import norm

# Functions for Step 1
#---------------------
def find_cylinder_frame(center_circle) :

  # Find the frame of the cylinder
  #
  # INPUT
  # - center_circle : 2*array 3*1 coordinates of the first and second circles of the cylinder
  #
  # OUTPUT
  # - cylinder_frame : array 3*3 ortonormal frame for the cylinder

  vect = center_circle[1] - center_circle[0]
  unit_vect = vect / norm(vect) # z axis du cylindre

  # Make some vector not in the same direction as vect_U
  not_unit_vect = np.array([1, 0, 0])
  if (unit_vect == [1,0,0]).all() or (unit_vect == [-1,0,0]).all():
    not_unit_vect = np.array([0, 1, 0])

  # Make a normalized vector perpendicular to vect_U
  n1 = np.cross(unit_vect, not_unit_vect)/norm(np.cross(unit_vect, not_unit_vect)) # notre y par exemple

  # Make unit vector perpendicular to v and n1
  n2 = np.cross(n1, unit_vect) # notre x par exemple

  return np.array([n2,n1, unit_vect])

def switch_frame(point, rotation_matrix, vect) :

  # Express point in a new frame
  #
  # INPUT
  # - point : array 3*1 coordinates of the point
  # - rotation_matrix : array 3*3 rotation matrix to change frame
  # - vect : 3*1 transition vector to change frame
  #
  # OUTPUT
  # - point_new_frame : array 3*1 coordinates of the point in the nex frame
  # ----------------------------------
  # transformation_matrix = np.array([[rotation_matrix[0][0], rotation_matrix[0][1], rotation_matrix[0][2], vect[0]],
  #                                   [rotation_matrix[1][0], rotation_matrix[1][1], rotation_matrix[1][2], vect[1]],
  #                                   [rotation_matrix[2][0], rotation_matrix[2][1], rotation_matrix[2][2], vect[2]],
  #                                   [0, 0, 0, 1]])
  # ----------------------------------

  return vect + np.dot(rotation_matrix, point)

def transpose_switch_frame(point, rotation_matrix, vect) :

  # Express point in its previous frame
  #
  # INPUT
  # - point : array 3*1 coordinates of the point
  # - rotation_matrix : array 3*3 rotation matrix to change frame
  # - vect : 3*1 transition vector to change frame
  #
  # OUTPUT
  # - point_previous_frame : array 3*1 coordinates of the point in its previous frame
  # ----------------------------------
  # transformation_matrix = np.array([[rotation_matrix[0][0], rotation_matrix[0][1], rotation_matrix[0][2], vect_transition[0]],
  #                                 [rotation_matrix[1][0], rotation_matrix[1][1], rotation_matrix[1][2], vect_transition[1]],
  #                                 [rotation_matrix[2][0], rotation_matrix[2][1], rotation_matrix[2][2], vect_transition[2]],
  #                                 [0, 0, 0, 1]])
  # ----------------------------------

  vect_transition = np.dot(rotation_matrix, vect)

  return vect_transition + np.dot(rotation_matrix, point)

# Functions for Step 2
#---------------------

def point_inside_cylinder(P, S, radius):
  # Exception if P or S are in the cylinder
  #
  # INPUT
  # - P : array 3*1 position of the first point
  # - S : array 3*1 position of the second point
  # - radius : radius of the cylinder
  #
  # OUTPUT
  # - point_inside : bool, True if P or S are in the cylinder, False otherwise

    if np.linalg.norm(P[:2]) < radius or np.linalg.norm(S[:2]) < radius :
        return True
    else:
        return False

def find_tangent_points_xy(p0, p1, r) :

  # Compute xy coordinates of v1 and v2
  #
  # INPUT
  # - p0 : array 3*1 position of the first point
  # - p1 : array 3*1 position of the second point
  # - r : radius of the cylinder * side
  #
  # OUTPUT
  # - v1 = [v1_x, v1_y, 0] : array 3*1 position of the first obstacle tangent point
  # - v2 = [v2_x, v2_y, 0] : array 3*1 position of the second obstacle tangent point

  p0_x2y2 = p0[0] ** 2 + p0[1] ** 2
  p1_x2y2 = p1[0] ** 2 + p1[1] ** 2

  if p0[0]**2+p0[1]**2-r**2 < 0 :
    v1_x = (p0[0]*r**2 + r*p0[1]*np.sqrt(0))/p0_x2y2 # + c[0]
    v1_y = (p0[1]*r**2 - r*p0[0]*np.sqrt(0))/p0_x2y2 # + c[1]
  else :
    v1_x = (p0[0]*r**2 + r*p0[1]*np.sqrt(p0[0]**2+p0[1]**2-r**2))/p0_x2y2 # + c[0]
    v1_y = (p0[1]*r**2 - r*p0[0]*np.sqrt(p0[0]**2+p0[1]**2-r**2))/p0_x2y2 # + c[1]

  if p1[0]**2+p1[1]**2-r**2 < 0 :
    v2_x = (p1[0]*r**2 - r*p1[1]*np.sqrt(0))/p1_x2y2 # + c[0]
    v2_y = (p1[1]*r**2 + r*p1[0]*np.sqrt(0))/p1_x2y2 # + c[1]
  else :
    v2_x = (p1[0]*r**2 - r*p1[1]*np.sqrt(p1[0]**2+p1[1]**2-r**2))/p1_x2y2 # + c[0]
    v2_y = (p1[1]*r**2 + r*p1[0]*np.sqrt(p1[0]**2+p1[1]**2-r**2))/p1_x2y2 # + c[1]

  return [v1_x, v1_y, 0], [v2_x, v2_y, 0]

def compute_length_v1_v2_xy(v1,v2, r) :

  # Compute xy coordinates of segment lengths in plane
  #
  # INPUT
  # - v1 : array 3*1 position of the first obstacle tangent point
  # - v2 : array 3*1 position of the second obstacle tangent point
  # - r : radius of the cylinder * side
  #
  # OUTPUT
  # - ||v1v2||(x,y) : xy coordinates of segment lengths in plane

  if r == 0:
    raise ValueError("Please choose an other radius, positive or negative are accepted. You musn't have r=0")

  return np.absolute(r*np.arccos(1.0-((v1[0]-v2[0])**2+(v1[1]-v2[1])**2)/(2*r**2)))

def z_coordinates_v1_v2(v1,v2,v1_v2_length_xy, origin_point, final_point) :

  # Compute z coordinates of v1 and v2
  #
  # INPUT
  # - v1 : array 3*1 position of the first obstacle tangent point
  # - v2 : array 3*1 position of the second obstacle tangent point
  # - v1_v2_length_xy : xy coordinates of segment lengths in plane
  # - origin_point : array 3*1 position of the first point
  # - final_point : array 3*1 position of the second point
  #
  # OUTPUT
  # - v1_z = z coordinate of v1
  # - v2_z = z coordinate of v2

  # Calculate the length of origin_point,v1(x,y) and v2,final_point(x,y)
  origin_point_v1_length_xy = np.sqrt((v1[0]-origin_point[0])**2 + (v1[1]-origin_point[1])**2)
  v2_final_point_length_xy = np.sqrt((final_point[0]-v2[0])**2 + (final_point[1]-v2[1])**2)

  v1_z= origin_point[2]+(((final_point[2]-origin_point[2])*origin_point_v1_length_xy)/
    (origin_point_v1_length_xy + v1_v2_length_xy + v2_final_point_length_xy))
  v2_z= final_point[2]-(((final_point[2]-origin_point[2])*v2_final_point_length_xy)/
    (origin_point_v1_length_xy + v1_v2_length_xy + v2_final_point_length_xy))

  return v1_z, v2_z

def find_tangent_points(p0, p1, r) :

   # Compute xyz coordinates of v1 and v2
   #
   # INPUT
   # - p0 : array 3*1 position of the first point
   # - p1 : array 3*1 position of the second point
   # - r : radius of the cylinder * side
   #
   # OUTPUT
   # - v1 = [v1_x, v1_y, v1_z] : array 3*1 position of the first obstacle tangent point
   # - v2 = [v2_x, v2_y, v2_z] : array 3*1 position of the second obstacle tangent point

   v1, v2 = find_tangent_points_xy(p0, p1, r)
   v1_v2_length_xy = compute_length_v1_v2_xy(v1, v2, r)
   v1[2], v2[2] = z_coordinates_v1_v2(v1,v2,v1_v2_length_xy, p0, p1)

   return v1, v2

def find_tangent_points_iterative_method(P_U_cylinder_frame, S_V_cylinder_frame, r_V, r_U, rotation_matrix_UV, origin_V_in_U_frame, origin_U_in_V_frame ) :

   # Compute xyz coordinates of Q, G, H and T using iterative method
   #
   # INPUT
   # - P_U_cylinder_frame : array 3*1 position of the first point in U cylinder frame
   # - S_V_cylinder_frame : array 3*1 position of the second point in V cylinder frame
   # - r_V : radius of the cylinder U * side_U
   # - r_U : radius of the cylinder V * side_V
   # - rotation_matrix_UV : array 3*3 rotation matrix to change frame (U --> V)
   # - origin_V_in_U_frame : array 3*1 coordinates of the center of the cylinder V in U cylinder frame
   # - origin_U_in_V_frame : array 3*1 coordinates of the center of the cylinder U in V cylinder frame
   #
   # OUTPUT
   # - Q0 : array 3*1 position of the first obstacle tangent point (in V cylinder frame)
   # - G0 : array 3*1 position of the second obstacle tangent point (in V cylinder frame)
   # - H0 : array 3*1 position of the third obstacle tangent point (in U cylinder frame)
   # - T0 : array 3*1 position of the fourth obstacle tangent point (in U cylinder frame)

   # v1_V est notre tangent point H0, v2_V est T, on fait dans le repere V
   H0, T0 = find_tangent_points(origin_U_in_V_frame, S_V_cylinder_frame, r_V)
   ecart_H0_H1 = [1,1,1]

   while (abs(ecart_H0_H1[0]) > 0.000001 or abs(ecart_H0_H1[1]) > 0.000001 or abs(ecart_H0_H1[2]) > 0.000001) :

    # On passe notre H0 dans le ref du cylindre U --> h0
    h0 = switch_frame(H0, rotation_matrix_UV, origin_V_in_U_frame)

    # On fait maintenant le calcul de Q et G, soit v1_U et v2_U
    Q0, G0 = find_tangent_points(P_U_cylinder_frame, h0,r_U)

    # Notre G est v1_U, on veut g dans le frame du cylindre V
    g0 = switch_frame(G0, np.transpose(rotation_matrix_UV), origin_U_in_V_frame)

    # On calcule v1_V et v2_V à partir de g0
    H1, T1 = find_tangent_points(g0, S_V_cylinder_frame,r_V)

    ecart_H0_H1 = np.array(H1)-np.array(H0)

    H0=H1
    T0=T1

   return Q0, G0, H0, T0

# Functions for Step 3
#----------------------

def determine_if_tangent_points_inactive_single_cylinder(v1,v2, r) :

  # Determine if tangent points v1 and v2 are inactive
  #
  # /!\ Differences with the B.A. Garner and M.G. Pandy paper !
  #   if Det < 0 : orientation is clockwise
  #   so for a side right-handed (side = 1), we need actived tangent points
  #   so, if Det*r < 0 ==> determine_if_tangent_points_inactive = False
  #   (and not "True" as originally presented in the paper)
  #
  # INPUT
  # - v1 : array 3*1 position of the first obstacle tangent point
  # - v2 : array 3*1 position of the second obstacle tangent point
  # - r : radius of the cylinder * side
  #
  # OUTPUT
  # - obstacle_tangent_point_inactive: bool True if tangent points are inactive --> Muscle path is straight line from origin point to final point
  #                                     False if tangent points are active --> Muscle passes by the two tangent points

  if (r*(v1[0]*v2[1] - v1[1]*v2[0])<0) :
    return False
  else :
    return True

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

def segment_length_double_cylinder(Q_G_inactive, H_T_inactive, P, S, P_U_cylinder_frame, P_V_cylinder_frame, S_U_cylinder_frame, S_V_cylinder_frame, Q, G, H, T, r_U, r_V, cylinder_frame_U, cylinder_frame_V, U_origin, V_origin) :

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
   # - cylinder_frame_U : array 3*3 ortonormal frame for the cylinder U
   # - cylinder_frame_V : array 3*3 ortonormal frame for the cylinder V
   # - U_origin : array 3*1 coordinates of the center of the cylinder
   # - V_origin : array 3*1 coordinates of the center of the cylinder
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

       G_H_length = norm(switch_frame(H, np.transpose(cylinder_frame_V), V_origin) - switch_frame(G, np.transpose(cylinder_frame_U), U_origin))

       segment_length = norm(Q - np.array(P_U_cylinder_frame)) + Q_G_length + G_H_length + H_T_length_xy + norm(np.array(S_V_cylinder_frame) - T)

   return segment_length

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

# Functions for Plot
#---------------------------

def data_cylinder(center_circle_1, center_circle_2, cylinder_frame, radius, num_points = 100) :

  # Compute datas for plot the cylinder
  # The cylinder is charaterized by coordinates of his two circle face and his radius
  #
  # INPUT
  # - center_circle_2 : array 3*1 coordinates of the first circle of the cylinder
  # - center_circle_2 : array 3*1 coordinates of the second circle of the cylinder
  # - cylinder_frame : array 3*3 local frame of the cylinder
  # - radius : radius of the cylinder
  # - num_points : int number of points for representation (default 100)
  #
  # OUTPUT
  # - X, Y, Z :  array nm_point*num_point coordinates of points for the representation of the cylinder

  # Create a unit vector in direction of axis
  v_cylinder=center_circle_2-center_circle_1

  n2,n1,v_unit = cylinder_frame

  # Surface ranges over t from 0 to length of axis and 0 to 2*pi
  t = np.linspace(0, norm(v_cylinder), num_points)
  theta = np.linspace(0, 2 * np.pi, num_points)
  t, theta = np.meshgrid(t, theta)

  # Generate coordinates for surface
  X, Y, Z = [center_circle_1[i] + v_unit[i] * t + radius * np.sin(theta) * n1[i] + radius * np.cos(theta) * n2[i] for i in [0, 1, 2]]
  return X, Y, Z

def data_semi_circle(v1, v2, cylinder_origin, cylinder_frame, r, num_points=100) :

  # Compute datas for plot the semi-circle between bounding fixed tangent points v1 and v2
  #
  # INPUT
  # - v1 : array 3*1 position of the first obstacle tangent point
  # - v2 : array 3*1 position of the second obstacle tangent point
  # - cylinder_origin : array 3*1 coordinates of the center of the cylinder
  # - cylinder_frame : array 3*3 local frame of the cylinder
  # - r : radius of the cylinder
  # - num_points : int number of points for representation (default 100)
  #
  # OUTPUT
  # - semi_circle_points : array nm_point*n3 coordinates of points for the representation of the semi-circle

  # Change frame
  v1 = transpose_switch_frame(v1, cylinder_frame, [0,0,0] - cylinder_origin)
  v2 = transpose_switch_frame(v2, cylinder_frame, [0,0,0] - cylinder_origin)
  c = np.array([0,0, (v1[2]+v2[2])/2])

  # Calculation of the normal vect of plan def by v1, v2 and c1
  norm = np.cross(v1 - c, v2 - c)
  norm /= np.linalg.norm(norm)

  # Calculate the angle between v1 and v2
  angle = np.arccos(np.dot((v1 - c) / np.linalg.norm(v1 - c), (v2 - c) / np.linalg.norm(v2 - c)))

  # Calculate points of the semi-circle
  theta = np.linspace(0, angle, num_points)
  semi_circle_points = c + r * np.cos(theta)[:, np.newaxis] * (v1 - c) / np.linalg.norm(v1 - c) + \
                        r * np.sin(theta)[:, np.newaxis] * np.cross(norm, (v1 - c) / np.linalg.norm(v1 - c))

  for i in range (len(semi_circle_points)) :
    semi_circle_points[i] = switch_frame(semi_circle_points[i], np.transpose(cylinder_frame), cylinder_origin)

  return semi_circle_points

def plot_one_cylinder_obstacle(origin_point, final_point, center_circle, radius, v1, v2, obstacle_tangent_point_inactive, segment_length, cylinder_origin, cylinder_frame) :

   # Plot the representation of the single-cylinder obstacle-set algorithm
   #
   # INPUT
   # - origin_point : array 3*1 position of the first point
   # - final_point : array 3*1 position of the second point
   # - center_circle : 2*array 3*1 coordinates of the first and second circles of the cylinder
   # - radius : radius of the cylinder
   # - v1 : array 3*1 position of the first obstacle tangent point
   # - v2 : array 3*1 position of the second obstacle tangent point
   # - obstacle_tangent_point_inactive : bool determine if v1 and v1 or inactive (True) or not (False)
   # - segment_length : length of path segments
   # - cylinder_origin : array 3*1 coordinates of the center of the cylinder
   # - cylinder_frame : array 3*3 local frame of the cylinder
   #
   # OUTPUT
   # - None : Plot axis, cylinder, points and muscle path

   # Set figure
   fig = plt.figure("Single Cylinder Wrapping")
   ax = fig.add_subplot(111, projection='3d')

   # Bouding-fixed tangent point
   ax.scatter(*origin_point, color='g', label="Origin point")
   ax.scatter(*final_point, color='b', label="Final point")

   #Obstacle tangent points
   ax.scatter(*v1, color='r', label="v1")
   ax.scatter(*v2, color='r', label="v2")

   # Cylinder
   Xc,Yc,Zc = data_cylinder(center_circle[0], center_circle[1], cylinder_frame, radius )
   ax.plot_surface(Xc, Yc, Zc, alpha=0.5)
   ax.plot(*zip(center_circle[0], center_circle[1]), color = 'k')

   if (obstacle_tangent_point_inactive == True) : # Muscle path is straight line from origin point to final point
    ax.plot(*zip(origin_point, final_point), color='r')
   else :
    # Semi-circle between v1 and v2
    semi_circle_points = data_semi_circle(v1,v2,cylinder_origin, cylinder_frame,radius, 100)
    ax.plot(semi_circle_points[:, 0], semi_circle_points[:, 1], semi_circle_points[:, 2])

    ax.plot(*zip(origin_point, v1), color='g')
    # ax.plot(*zip(v1, v2), color='r')
    ax.plot(*zip(v2, final_point), color='b')

   # Set graph
   ax.set_xlabel("X")
   ax.set_ylabel("Y")
   ax.set_zlabel("Z")
   ax.grid(True)

   # Set ax limit
   ax.set_xlim(-5,5)
   ax.set_ylim(-5,5)
   ax.set_zlim(-5,5)

   plt.title("Single Cylinder Wrapping")
   plt.legend()

   plt.show()

def plot_double_cylinder_obstacle(P, S, center_circle_U, center_circle_V, radius_U, radius_V, Q, G, H, T, cylinder_frame_U, cylinder_frame_V, U_origin, V_origin, Q_G_inactive, H_T_inactive ) :

   # Plot the representation of the double-cylinder obstacle-set algorithm
   #
   # INPUT
   # - P : array 3*1 position of the first point
   # - S : array 3*1 position of the second point
   # - center_circle_U : 2*array 3*1 coordinates of the first and second circles of the cylinder U
   # - center_circle_V : 2*array 3*1 coordinates of the first and second circles of the cylinder V
   # - radius_U : radius of the cylinder U
   # - radius_V : radius of the cylinder V
   # - Q : array 3*1 position of the first obstacle tangent point (in conventional frame)
   # - G : array 3*1 position of the second obstacle tangent point (in conventional frame)
   # - H : array 3*1 position of the third obstacle tangent point (in conventional frame)
   # - T : array 3*1 position of the fourth obstacle tangent point (in conventional frame)
   # - U_origin : array 3*1 coordinates of the center of the cylinder U
   # - V_origin : array 3*1 coordinates of the center of the cylinder V
   # - Q_G_inactive : bool determine if Q and G or inactive (True) or not (False)
   # - H_T_inactive : bool determine if H and T or inactive (True) or not (False)
   #
   # OUTPUT
   # - None : Plot axis, cylinder, points and muscle path

   # Set figure
   fig = plt.figure("Double Cylinder Wrapping")
   ax = fig.add_subplot(111, projection='3d')

   # Bouding-fixed tangent point
   ax.scatter(*P, color='g', label="Origin point")
   ax.scatter(*S, color='b', label="Final point")

   #Obstacle tangent points
   ax.scatter(*Q, color='r', label="Q")
   ax.scatter(*G, color='r', label="G")
   ax.scatter(*H, color='r', label="H")
   ax.scatter(*T, color='r', label="T")

   # 1st Cylinder
   Xcu,Ycu,Zcu = data_cylinder(center_circle_U[0], center_circle_U[1], cylinder_frame_U, radius_U)
   ax.plot_surface(Xcu, Ycu, Zcu, alpha=0.5)
   ax.plot(*zip(center_circle_U[0], center_circle_U[1]), color = 'k')

   # 2nd Cylinder
   Xcv,Ycv,Zcv = data_cylinder(center_circle_V[0], center_circle_V[1], cylinder_frame_V, radius_V)
   ax.plot_surface(Xcv, Ycv, Zcv, alpha=0.5)
   ax.plot(*zip(center_circle_V[0], center_circle_V[1]), color = 'k')


   if Q_G_inactive and H_T_inactive: # Muscle path is straight line from origin_point to final_point
       ax.plot(*zip(P, S), color='r')

   elif Q_G_inactive: # single cylinder algorithm with V cylinder
    # Semi-circle between H and T
    semi_circle_points = data_semi_circle(H,T,V_origin, cylinder_frame_V, radius_V, 100)
    ax.plot(semi_circle_points[:, 0], semi_circle_points[:, 1], semi_circle_points[:, 2])

    ax.plot(*zip(P, H), color='g')
    ax.plot(*zip(T, S), color='b')

   elif H_T_inactive: # single cylinder algorithm with U cylinder
    # Semi-circle between Q and G
    center_circle=np.array([0,0, (Q[2]+G[2])/2])
    semi_circle_points = data_semi_circle(Q,G, U_origin, cylinder_frame_U,radius_U, 100)
    ax.plot(semi_circle_points[:, 0], semi_circle_points[:, 1], semi_circle_points[:, 2])

    ax.plot(*zip(P, Q), color='g')
    ax.plot(*zip(G, S), color='b')

   else: # double cylinder

    # Semi-circle between H and T
    center_circle=np.array([0,0, (H[2]+T[2])/2])
    semi_circle_points = data_semi_circle(H,T,V_origin, cylinder_frame_V,radius_V, 100)
    ax.plot(semi_circle_points[:, 0], semi_circle_points[:, 1], semi_circle_points[:, 2])

    # Semi-circle between Q and G
    center_circle=np.array([0,0, (Q[2]+G[2])/2])
    semi_circle_points = data_semi_circle(Q,G, U_origin, cylinder_frame_U,radius_U, 100)
    ax.plot(semi_circle_points[:, 0], semi_circle_points[:, 1], semi_circle_points[:, 2])

    ax.plot(*zip(P, Q), color='g')
    ax.plot(*zip(G, H), color='b')
    ax.plot(*zip(T, S), color='b')

   # Set graph
   ax.set_xlabel("X")
   ax.set_ylabel("Y")
   ax.set_zlabel("Z")
   ax.grid(True)

   # Set ax limit
   ax.set_xlim(-5,5)
   ax.set_ylim(-5,5)
   ax.set_zlim(-5,5)

   plt.title("Double Cylinder Wrapping")
   plt.legend()

   plt.show()

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
   S =[0,4,2] # final_point

   # Points for 1st cylinder
   center_circle_U = [np.array([2,0,0]),np.array([-2,0,0])]
   radius_U = 1
   side_U = 1 # Choose 1 for the right-handed side, -1 for the left-handed side (with respect to the z-axis)

   # Points for 2nd cylinder
   center_circle_V = [np.array([0,1.5,-4]),np.array([0,1.5,4])]
   radius_V = 1
   side_V = 1 # Choose 1 for the right-handed side, -1 for the left-handed side (with respect to the z-axis)

   show_details = True
   single_cylinder = False # with U cylinder
   double_cylinder = True # with U and V cylinders

   # Other inputs -----------------------------------------
   # Nothing to change he
   U_origin = (center_circle_U[1] + center_circle_U[0]) / 2
   cylinder_frame_U = find_cylinder_frame(center_circle_U)

   V_origin = (center_circle_V[1] + center_circle_V[0]) / 2
   cylinder_frame_V = find_cylinder_frame(center_circle_V)

   # Rotation matrix UV
   rotation_matrix_UV = np.dot(np.transpose(cylinder_frame_V), cylinder_frame_U)

   if (single_cylinder) :
    # --------------------------------------
    # Single cylinder obstacle set algorithm
    # --------------------------------------
    v1, v2, obstacle_tangent_point_inactive, segment_length = single_cylinder_obstacle_set_algorithm(P,S, radius_U, side_U, U_origin, cylinder_frame_U)

    # ----
    # Plot
    # ----
    plot_one_cylinder_obstacle(P,S, center_circle_U, radius_U, v1, v2, obstacle_tangent_point_inactive, segment_length, U_origin, cylinder_frame_U)

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

    Q, G, H, T, Q_G_inactive, H_T_inactive, segment_length = double_cylinder_obstacle_set_algorithm(P, S, U_origin, cylinder_frame_U, radius_U, side_U, V_origin, cylinder_frame_V, radius_V, side_V, rotation_matrix_UV)

    # ----
    # Plot
    # ----
    plot_double_cylinder_obstacle(P, S, center_circle_U, center_circle_V, radius_U, radius_V, Q, G, H, T, cylinder_frame_U, cylinder_frame_V, U_origin, V_origin, Q_G_inactive, H_T_inactive )

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