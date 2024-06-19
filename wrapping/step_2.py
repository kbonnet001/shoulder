import numpy as np
from scipy.linalg import norm
from wrapping.step_1 import *
from wrapping.step_3 import determine_if_tangent_points_inactive_single_cylinder
from wrapping.step_4 import *

# Functions for Step 2
#---------------------

def point_inside_cylinder(P, radius, epsilon = 0.00095):
  """Exception if P or S are in the cylinder

  INPUT
  - P : array 3*1 position of the first point
  - S : array 3*1 position of the second point
  - radius : radius of the cylinder

  OUTPUT
  - point_inside : bool, True if P or S are in the cylinder, False otherwise"""

  if np.linalg.norm(P[:2]) < radius - epsilon:
      return True
  else:
      return False

def point_tangent_inside_cylinder(point, Cylinder_1, Cylinder_2, epsilon=0.00095):
  """Exception if P or S are in the cylinder

  INPUT
  - P : array 3*1 position of the first point
  - S : array 3*1 position of the second point
  - radius : radius of the cylinder

  OUTPUT
  - point_inside : bool, True if P or S are in the cylinder, False otherwise"""

  point = switch_frame_UV(point, Cylinder_1.matrix, Cylinder_2.matrix)

  if np.linalg.norm(point[:2]) < Cylinder_2.radius - epsilon:
      return True
  else:
      return False

def find_tangent_points_xy(p0, p1, r) :

  """Compute xy coordinates of v1 and v2

  INPUT
  - p0 : array 3*1 position of the first point
  - p1 : array 3*1 position of the second point
  - r : radius of the cylinder * side

  OUTPUT
  - v1 = [v1_x, v1_y, 0] : array 3*1 position of the first obstacle tangent point
  - v2 = [v2_x, v2_y, 0] : array 3*1 position of the second obstacle tangent point"""

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

  """Compute xy coordinates of segment lengths in plane

  INPUT
  - v1 : array 3*1 position of the first obstacle tangent point
  - v2 : array 3*1 position of the second obstacle tangent point
  - r : radius of the cylinder * side

  OUTPUT
  - ||v1v2||(x,y) : xy coordinates of segment lengths in plane"""

  if r == 0:
    raise ValueError("Please choose an other radius, positive or negative are accepted. You musn't have r=0")

  # if 2.5 > ((v1[0]-v2[0])**2+(v1[1]-v2[1])**2)/(2*r**2) > 2.0 : #2.5
  #   print("((v1[0]-v2[0])**2+(v1[1]-v2[1])**2)/(2*r**2) = ", ((v1[0]-v2[0])**2+(v1[1]-v2[1])**2)/(2*r**2))
  #   return np.absolute(r*np.arccos(1.0 - 2.0))
  # return np.absolute(r*np.arccos(1.0-((v1[0]-v2[0])**2+(v1[1]-v2[1])**2)/(2*r**2)))
  # Coordonnées des points A et B
  # x1, y1 = v1[0], v1[1]
  # x2, y2 = v2[0], v2[1]
  
  # # Calculer les angles polaires des points A et B par rapport au centre (0,0)
  # theta_A = np.arctan2(y1, x1)
  # theta_B = np.arctan2(y2, x2)
  
  # # Calculer la différence d'angle
  # delta_theta = theta_B - theta_A
  
  # # Ajuster l'angle pour qu'il soit dans l'intervalle [0, 2*pi]
  # if delta_theta < 0:
  #     delta_theta += 2 * np.pi
  
  # # Calculer la longueur de l'arc
  # arc_length = r * delta_theta
  
  # return arc_length
  
  # code similaire à celui du papier mais plus detaille
  # Coordonnées des points Q et T
  Qx, Qy = v1[:2]
  Tx, Ty = v2[:2]
  
  # Calculer la différence entre les coordonnées
  diff_x = Qx - Tx
  diff_y = Qy - Ty
  
  # Calculer le carré des différences
  diff_squared = diff_x**2 + diff_y**2
  
  # Calculer la valeur à l'intérieur du arc cosinus
  value = 1.0 - diff_squared / (2 * r**2)
  
  # Assurer que la valeur est dans l'intervalle [-1, 1]
  value = np.clip(value, -1, 1)
  
  # Calculer l'arc cosinus
  arc_cosine = np.arccos(value)
  
  # Multiplier par R et prendre la valeur absolue
  result = np.absolute(r * arc_cosine)
  
  return result
  
def angle_between_points(v1, v2):
    # Coordonnées des points A et B
    x1, y1 = v1[:2]
    x2, y2 = v2[:2]
    
    # Calculer les angles polaires des points A et B par rapport au centre (0,0)
    theta_A = np.arctan2(y1, x1)
    theta_B = np.arctan2(y2, x2)
    
    # Calculer la différence d'angle
    delta_theta = theta_B - theta_A
    
    # Ajuster l'angle pour qu'il soit dans l'intervalle [0, 2*pi]
    if delta_theta < 0:
        delta_theta += 2 * np.pi
    
    return delta_theta
  
def arc_length(v1, v2, r):
    # Coordonnées des points A et B
    x1, y1 = v1[:2]
    x2, y2 = v2[:2]
    
    # Calculer les angles polaires des points A et B par rapport au centre (0,0)
    theta_A = np.arctan2(y1, x1)
    theta_B = np.arctan2(y2, x2)
    
    # Calculer la différence d'angle
    delta_theta = theta_B - theta_A
    
    # Ajuster l'angle pour qu'il soit dans l'intervalle [0, 2*pi]
    if delta_theta < 0:
        delta_theta += 2 * np.pi
    
    # Calculer la longueur de l'arc
    arc_length = r * delta_theta
    
    return arc_length


def z_coordinates_v1_v2(v1,v2,v1_v2_length_xy, origin_point, final_point) :

  """Compute z coordinates of v1 and v2

  INPUT
  - v1 : array 3*1 position of the first obstacle tangent point
  - v2 : array 3*1 position of the second obstacle tangent point
  - v1_v2_length_xy : xy coordinates of segment lengths in plane
  - origin_point : array 3*1 position of the first point
  - final_point : array 3*1 position of the second point

  OUTPUT
  - v1_z = z coordinate of v1
  - v2_z = z coordinate of v2"""

  # Calculate the length of origin_point,v1(x,y) and v2,final_point(x,y)
  origin_point_v1_length_xy = np.sqrt((v1[0]-origin_point[0])**2 + (v1[1]-origin_point[1])**2)
  v2_final_point_length_xy = np.sqrt((final_point[0]-v2[0])**2 + (final_point[1]-v2[1])**2)

  v1_z= origin_point[2]+(((final_point[2]-origin_point[2])*origin_point_v1_length_xy)/
    (origin_point_v1_length_xy + v1_v2_length_xy + v2_final_point_length_xy))
  v2_z= final_point[2]-(((final_point[2]-origin_point[2])*v2_final_point_length_xy)/
    (origin_point_v1_length_xy + v1_v2_length_xy + v2_final_point_length_xy))

  return v1_z, v2_z

def find_tangent_points(p0, p1, r) :

   """Compute xyz coordinates of v1 and v2

   INPUT
   - p0 : array 3*1 position of the first point
   - p1 : array 3*1 position of the second point
   - r : radius of the cylinder * side

   OUTPUT
   - v1 = [v1_x, v1_y, v1_z] : array 3*1 position of the first obstacle tangent point
   - v2 = [v2_x, v2_y, v2_z] : array 3*1 position of the second obstacle tangent point"""

   v1, v2 = find_tangent_points_xy(p0, p1, r)
   v1_v2_length_xy = compute_length_v1_v2_xy(v1, v2, r)
   v1[2], v2[2] = z_coordinates_v1_v2(v1,v2,v1_v2_length_xy, p0, p1)

   return v1, v2

#################
def find_tangent_points_iterative_method(P, S, P_U_cylinder_frame,P_V_cylinder_frame, S_U_cylinder_frame, S_V_cylinder_frame, r_V, r_U, matrix_U, matrix_V) :

  """Compute xyz coordinates of Q, G, H and T using iterative method

  INPUT
  - P_U_cylinder_frame : array 3*1 position of the first point in U cylinder frame
  - S_V_cylinder_frame : array 3*1 position of the second point in V cylinder frame
  - r_V : radius of the cylinder U * side_U
  - r_U : radius of the cylinder V * side_V
  - rotation_matrix_UV : array 4*4 rotation matrix and vect to change frame (U --> V)

  OUTPUT
  - Q0 : array 3*1 position of the first obstacle tangent point (in V cylinder frame)
  - G0 : array 3*1 position of the second obstacle tangent point (in V cylinder frame)
  - H0 : array 3*1 position of the third obstacle tangent point (in U cylinder frame)
  - T0 : array 3*1 position of the fourth obstacle tangent point (in U cylinder frame)"""

  # cylindre U
  Q1, G1 = find_tangent_points(P_U_cylinder_frame, S_V_cylinder_frame, r_U)
  H1, T1 = [0,0,0], [0,0,0]
  Q0, G0 = [0,0,0], [0,0,0]
  H0, T0 = [0,0,0], [0,0,0]
  Q1_G1_inactive = False
  H1_T1_inactive = False
  ecart_length = 1
  segment_length_1 = 100
  # k = 0
  while Q1_G1_inactive == False and H1_T1_inactive == False and ecart_length > 0 and np.isnan(np.array(Q1)).any()==False and np.isnan(np.array(G1)).any()==False and np.isnan(np.array(H1)).any()==False and np.isnan(np.array(T1)).any() == False:
      Q0, G0 = Q1, G1 # 166 que ecart et echec avec les break
      H0, T0 = H1, T1
      segment_length_0 = segment_length_1

      g0 = switch_frame_UV(H1, matrix_U, matrix_V)
      # g0 = transpose_switch_frame(G0, matrix_V)
      H1, T1 = find_tangent_points(g0, S_V_cylinder_frame, r_V)
      H1_T1_inactive = determine_if_tangent_points_inactive_single_cylinder(H1, T1, r_V)
      # if k != 0 and (H1_T1_inactive or np.isnan(np.array(H1)).any() or np.isnan(np.array(T1)).any()):
      if H1_T1_inactive or np.isnan(np.array(H1)).any() or np.isnan(np.array(T1)).any():
        break

      h0 = switch_frame_UV(H1, matrix_V, matrix_U)
      Q1, G1 = find_tangent_points(P_U_cylinder_frame, h0, r_U)
      Q1_G1_inactive = determine_if_tangent_points_inactive_single_cylinder(Q1, G1, r_U)
      if Q1_G1_inactive or np.isnan(np.array(Q1)).any() or np.isnan(np.array(G1)).any():
        break #laisser lui aue le while

      segment_length_1 = segment_length_double_cylinder(False, False, P, S, P_U_cylinder_frame, P_V_cylinder_frame, S_U_cylinder_frame, S_V_cylinder_frame, Q1, G1, H1, T1, r_U, r_V, matrix_U, matrix_V)
      ecart_length = segment_length_0 - segment_length_1
      k+=1

  return Q0, G0, H0, T0

#################

# Functions for Step 4
#----------------------

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
       G_H_length = norm(switch_frame(H, matrix_V) - switch_frame(G, matrix_U))

       segment_length = norm(Q - np.array(P_U_cylinder_frame)) + Q_G_length + G_H_length + H_T_length + norm(np.array(S_V_cylinder_frame) - T)

   return segment_length
 
def segment_length_double_cylinder2(Q_G_inactive, H_T_inactive, P, S, P_U_cylinder_frame, P_V_cylinder_frame, S_U_cylinder_frame, S_V_cylinder_frame, Q, G, H, T, r_U, r_V, matrix_U, matrix_V) :

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
  H_T_length_xy = arc_length(H,T,r_V)
  H_T_length = compute_length_v1_v2(H, T, H_T_length_xy)
  Q_G_length_xy = arc_length(Q, G, r_U)
  Q_G_length = compute_length_v1_v2(Q, G, Q_G_length_xy)

  # Compute segment lengths based on conditions
  if Q_G_inactive and H_T_inactive: # Muscle path is straight line from origin_point to final_point
      segment_length = norm(np.array(P) - np.array(S))

  elif Q_G_inactive: # single cylinder algorithm with V cylinder
      segment_length = norm(H - np.array(P_V_cylinder_frame)) + H_T_length + norm(np.array(S_V_cylinder_frame) - T)

  elif H_T_inactive: # single cylinder algorithm with U cylinder
      segment_length = norm(Q - np.array(P_U_cylinder_frame)) + Q_G_length + norm(np.array(S_U_cylinder_frame) - G)

  else: # double cylinder
      G_H_length = norm(switch_frame(H, matrix_V) - switch_frame(G, matrix_U))

      segment_length = norm(Q - np.array(P_U_cylinder_frame)) + Q_G_length + G_H_length + H_T_length + norm(np.array(S_V_cylinder_frame) - T)

  return segment_length