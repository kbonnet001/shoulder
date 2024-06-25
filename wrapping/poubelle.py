# import numpy as np
# from wrapping.step_1 import switch_frame_UV

# def point_inside_cylinder(P, radius, epsilon = 0.00095):
#   """Exception if P or S are in the cylinder

#   INPUT
#   - P : array 3*1 position of the first point
#   - S : array 3*1 position of the second point
#   - radius : radius of the cylinder

#   OUTPUT
#   - point_inside : bool, True if P or S are in the cylinder, False otherwise"""

#   if np.linalg.norm(P[:2]) < radius - epsilon:
#       return True
#   else:
#       return False

# def point_tangent_inside_cylinder(point, Cylinder_1, Cylinder_2, epsilon=0.00095):
#   """Exception if P or S are in the cylinder

#   INPUT
#   - P : array 3*1 position of the first point
#   - S : array 3*1 position of the second point
#   - radius : radius of the cylinder

#   OUTPUT
#   - point_inside : bool, True if P or S are in the cylinder, False otherwise"""

#   point = switch_frame_UV(point, Cylinder_1.matrix, Cylinder_2.matrix)

#   if np.linalg.norm(point[:2]) < Cylinder_2.radius - epsilon:
#       return True
#   else:
#       return False
  
# def angle_between_points(v1, v2):
#     # Coordonnées des points A et B
#     x1, y1 = v1[:2]
#     x2, y2 = v2[:2]
    
#     # Calculer les angles polaires des points A et B par rapport au centre (0,0)
#     theta_A = np.arctan2(y1, x1)
#     theta_B = np.arctan2(y2, x2)
    
#     # Calculer la différence d'angle
#     delta_theta = theta_B - theta_A
    
#     # Ajuster l'angle pour qu'il soit dans l'intervalle [0, 2*pi]
#     if delta_theta < 0:
#         delta_theta += 2 * np.pi
    
#     return delta_theta
  
# def arc_length(v1, v2, r):
#     # Coordonnées des points A et B
#     x1, y1 = v1[:2]
#     x2, y2 = v2[:2]
    
#     # Calculer les angles polaires des points A et B par rapport au centre (0,0)
#     theta_A = np.arctan2(y1, x1)
#     theta_B = np.arctan2(y2, x2)
    
#     # Calculer la différence d'angle
#     delta_theta = theta_B - theta_A
    
#     # Ajuster l'angle pour qu'il soit dans l'intervalle [0, 2*pi]
#     if delta_theta < 0:
#         delta_theta += 2 * np.pi
    
#     # Calculer la longueur de l'arc
#     arc_length = r * delta_theta
    
#     return arc_length

# def segment_length_double_cylinder2(Q_G_inactive, H_T_inactive, P, S, P_U_cylinder_frame, P_V_cylinder_frame, S_U_cylinder_frame, S_V_cylinder_frame, Q, G, H, T, r_U, r_V, matrix_U, matrix_V) :

#   # Compute length of path segments
#   #
#   # INPUT
#   # - Q_G_inactive : bool determine if Q and G or inactive (True) or not (False)
#   # - H_T_inactive : bool determine if H and T or inactive (True) or not (False)
#   # - P : array 3*1 position of the first point
#   # - S : array 3*1 position of the second point
#   # - P_U_cylinder_frame : array 3*1 position of the first point in U cylinder frame
#   # - P_V_cylinder_frame : array 3*1 position of the first point in V cylinder frame
#   # - S_U_cylinder_frame : array 3*1 position of the second point in U cylinder frame
#   # - S_V_cylinder_frame : array 3*1 position of the second point in V cylinder frame
#   # - Q : array 3*1 position of the first obstacle tangent point (in V cylinder frame)
#   # - G : array 3*1 position of the second obstacle tangent point (in V cylinder frame)
#   # - H : array 3*1 position of the third obstacle tangent point (in U cylinder frame)
#   # - T : array 3*1 position of the fourth obstacle tangent point (in U cylinder frame)
#   # - r_V : radius of the cylinder U * side_U
#   # - r_U : radius of the cylinder V * side_V
#   # - matrix_U : array 4*4 rotation_matrix and vect for cylinder U
#   # - matrix_V : array 4*4 rotation_matrix and vect for cylinder V
#   #
#   # OUTPUT
#   # - segment_length : length of path segments

#   # Compute lengths
#   H_T_length_xy = arc_length(H,T,r_V)
#   H_T_length = compute_length_v1_v2(H, T, H_T_length_xy)
#   Q_G_length_xy = arc_length(Q, G, r_U)
#   Q_G_length = compute_length_v1_v2(Q, G, Q_G_length_xy)

#   # Compute segment lengths based on conditions
#   if Q_G_inactive and H_T_inactive: # Muscle path is straight line from origin_point to final_point
#       segment_length = norm(np.array(P) - np.array(S))

#   elif Q_G_inactive: # single cylinder algorithm with V cylinder
#       segment_length = norm(H - np.array(P_V_cylinder_frame)) + H_T_length + norm(np.array(S_V_cylinder_frame) - T)

#   elif H_T_inactive: # single cylinder algorithm with U cylinder
#       segment_length = norm(Q - np.array(P_U_cylinder_frame)) + Q_G_length + norm(np.array(S_U_cylinder_frame) - G)

#   else: # double cylinder
#       G_H_length = norm(switch_frame(H, matrix_V) - switch_frame(G, matrix_U))

#       segment_length = norm(Q - np.array(P_U_cylinder_frame)) + Q_G_length + G_H_length + H_T_length + norm(np.array(S_V_cylinder_frame) - T)
#       print("H_T_length = ", H_T_length)

#   return segment_length

# def rotation_direction(self,matrix_initial, matrix_update):
#     """
#     Determine the direction of rotation around the Z axis between two transformation matrices.
    
#     INPUT
#     - matrix_initial : array 4*4 initial rotation and translation matrix
#     - matrix_update : array 4*4 updated rotation and translation matrix
    
#     OUTPUT
#     - 1 if the rotation around the Z axis is clockwise
#     - -1 if the rotation around the Z axis is counterclockwise
#     """
#     # Extract rotation parts from the transformation matrices
#     rot_initial = matrix_initial[:3, :3]
#     rot_update = matrix_update[:3, :3]
    
#     # Calculate the relative rotation matrix
#     rot_relative = rot_update @ np.linalg.inv(rot_initial)
    
#     # Convert the relative rotation matrix to Euler angles
#     euler_angles = R.from_matrix(rot_relative).as_euler('xyz', degrees=True)
    
#     # Extract the angle of rotation around the Z axis
#     angle_z = euler_angles[2]
    
#     # Determine the direction of rotation
#     if angle_z <= 0:
#         return 1  # Clockwise
#     else:
#         return -1  # Counterclockwise