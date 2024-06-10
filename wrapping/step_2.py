import numpy as np
from scipy.linalg import norm
from wrapping.step_1 import *
from wrapping.step_3 import determine_if_tangent_points_inactive_single_cylinder
# from wrapping.step_4 import segment_length_double_cylinder

# Functions for Step 2
#---------------------

def point_inside_cylinder(P, S, radius):
  """Exception if P or S are in the cylinder
  
  INPUT
  - P : array 3*1 position of the first point
  - S : array 3*1 position of the second point
  - radius : radius of the cylinder
  
  OUTPUT
  - point_inside : bool, True if P or S are in the cylinder, False otherwise"""

  if np.linalg.norm(P[:2]) < radius or np.linalg.norm(S[:2]) < radius :
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

  return np.absolute(r*np.arccos(1.0-((v1[0]-v2[0])**2+(v1[1]-v2[1])**2)/(2*r**2)))

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

def find_tangent_points_iterative_method(P_U_cylinder_frame, S_V_cylinder_frame, r_V, r_U, rotation_matrix_UV) :

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

  # v1_V est notre tangent point H0, v2_V est T, on fait dans le repere V
  H0, T0 = find_tangent_points(rotation_matrix_UV[0:3, 3], S_V_cylinder_frame, r_V)
  ecart_H0_H1 = [1,1,1]

  while (abs(ecart_H0_H1[0]) > 1e-4  or abs(ecart_H0_H1[1]) > 1e-4  or abs(ecart_H0_H1[2]) > 1e-4) :

    # On passe notre H0 dans le ref du cylindre U --> h0
    h0 = transpose_switch_frame(H0, rotation_matrix_UV) # ok à priori ...

    # On fait maintenant le calcul de Q et G, soit v1_U et v2_U
    Q0, G0 = find_tangent_points(P_U_cylinder_frame, h0,r_U)

    # Notre G est v1_U, on veut g dans le frame du cylindre V
    g0 = switch_frame(G0, rotation_matrix_UV)

    # On calcule v1_V et v2_V à partir de g0
    H1, T1 = find_tangent_points(g0, S_V_cylinder_frame,r_V)

    ecart_H0_H1 = np.array(H1)-np.array(H0)

    H0=H1
    T0=T1

  return Q0, G0, H0, T0
 
# def find_tangent_points_iterative_method2(P_U_cylinder_frame, S_U_cylinder_frame, S_V_cylinder_frame, r_V, r_U, rotation_matrix_UV) :

#   """Compute xyz coordinates of Q, G, H and T using iterative method
  
#   INPUT
#   - P_U_cylinder_frame : array 3*1 position of the first point in U cylinder frame
#   - S_V_cylinder_frame : array 3*1 position of the second point in V cylinder frame
#   - r_V : radius of the cylinder U * side_U
#   - r_U : radius of the cylinder V * side_V
#   - rotation_matrix_UV : array 4*4 rotation matrix and vect to change frame (U --> V)
  
#   OUTPUT
#   - Q0 : array 3*1 position of the first obstacle tangent point (in V cylinder frame)
#   - G0 : array 3*1 position of the second obstacle tangent point (in V cylinder frame)
#   - H0 : array 3*1 position of the third obstacle tangent point (in U cylinder frame)
#   - T0 : array 3*1 position of the fourth obstacle tangent point (in U cylinder frame)"""

#   # v1_V est notre tangent point H0, v2_V est T, on fait dans le repere V
#   Q0, G0 = find_tangent_points(P_U_cylinder_frame, S_U_cylinder_frame, r_U)

#     # Notre G est v1_U, on veut g dans le frame du cylindre V
#   g0 = transpose_switch_frame(G0, rotation_matrix_UV)

#   # On calcule v1_V et v2_V à partir de g0
#   H0, T0 = find_tangent_points(g0, S_V_cylinder_frame,r_V)

#   return Q0, G0, H0, T0

def find_tangent_points_iterative_method_0(P_U_cylinder_frame,P_V_cylinder_frame, S_U_cylinder_frame, S_V_cylinder_frame, r_V, r_U, rotation_matrix_VU, matrix_U, matrix_V) :

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

  # v1_V est notre tangent point H0, v2_V est T, on fait dans le repere V
  Q0, G0 = find_tangent_points(P_U_cylinder_frame, S_U_cylinder_frame, r_U)

  # while (abs(ecart_G0_G1[0]) > 1e-4  or abs(ecart_G0_G1[1]) > 1e-4  or abs(ecart_G0_G1[2]) > 1e-4) :

  # On passe notre H0 dans le ref du cylindre U --> h0
  # g0 = transpose_switch_frame(G0, rotation_matrix_VU) # ok à priori ...
  g0 = switch_frame(G0, matrix_U)
  g0 = transpose_switch_frame(G0, matrix_V)

  # On fait maintenant le calcul de Q et G, soit v1_U et v2_U
  H0, T0 = find_tangent_points(g0, S_V_cylinder_frame, r_V)
  
  # on veut le H en u
  h0 = switch_frame(H0, matrix_V)
  h0 = transpose_switch_frame(h0, matrix_U)

  # fait un wrapping U 
  Q0, G0 = find_tangent_points(P_U_cylinder_frame, h0, r_U)
  
  g0 = switch_frame(G0, matrix_U)
  g0 = transpose_switch_frame(G0, matrix_V)

  # On fait maintenant le calcul de Q et G, soit v1_U et v2_U
  H0, T0 = find_tangent_points(g0, S_V_cylinder_frame, r_V)
  
  return Q0, G0, H0, T0
 
def find_tangent_points_iterative_method_3(P_U_cylinder_frame,P_V_cylinder_frame, S_U_cylinder_frame, S_V_cylinder_frame, r_V, r_U, rotation_matrix_VU, matrix_U, matrix_V) :

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
  
  # on fait zrapping de v
  H0, T0 = find_tangent_points(P_V_cylinder_frame, S_V_cylinder_frame, r_V)
  
  # on veut le H en u
  h0 = switch_frame(H0, matrix_V)
  h0 = transpose_switch_frame(h0, matrix_U)

  # fait un wrapping U 
  Q0, G0 = find_tangent_points(P_U_cylinder_frame, h0, r_U)
  # on regarde si il y a vraiement un wrapping ou pas
  
  Q0_G0_inactive = determine_if_tangent_points_inactive_single_cylinder(G0, Q0, r_U)
  
  if Q0_G0_inactive : # il faut faire un wrapping 2 cylindres
    Q0, G0, H0, T0 = find_tangent_points_iterative_method_2(P_U_cylinder_frame,P_V_cylinder_frame, S_U_cylinder_frame, S_V_cylinder_frame, r_V, r_U, rotation_matrix_VU, matrix_U, matrix_V) 
  else : # faut pas faire, juste avec V
    Q0, G0 = [0,0,0], [0,0,0]
  
  return Q0, G0, H0, T0


def find_tangent_points_iterative_method_4(P, S, P_U_cylinder_frame,P_V_cylinder_frame, S_U_cylinder_frame, S_V_cylinder_frame, r_V, r_U, rotation_matrix_VU, matrix_U, matrix_V) :

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
  
  # on fait zrapping de v
  H1, T1 = find_tangent_points(P_V_cylinder_frame, S_V_cylinder_frame, r_V)
  
  # on veut le H en u
  h0 = switch_frame(H1, matrix_V)
  print("ho =" , h0)
  h0 = transpose_switch_frame(h0, matrix_U)
  print("ho 2 =" , h0)

  # fait un wrapping U 
  Q1, G1 = find_tangent_points(P_U_cylinder_frame, h0, r_U)
  print("Q1 et G1  = ", Q1, G1 )
  # on regarde si il y a vraiement un wrapping ou pas
  Q1_G1_inactive = determine_if_tangent_points_inactive_single_cylinder(G1, Q1, r_U)
  print("Q1_G1_inactive  0 = ", Q1_G1_inactive )
  segment_length_1 = 100
  ecart_length = 1
  
  if Q1_G1_inactive :
    Q1, G1 = find_tangent_points(P_U_cylinder_frame, S_V_cylinder_frame, r_U)
    
    # Q0, G0, H0, T0 = find_tangent_points_iterative_method_2(P_U_cylinder_frame,P_V_cylinder_frame, S_U_cylinder_frame, S_V_cylinder_frame, r_V, r_U, rotation_matrix_VU, matrix_U, matrix_V) 
    # while Q1_G1_inactive and (abs(ecart_Q0_Q1[0]) > value  or abs(ecart_Q0_Q1[1]) > value  or abs(ecart_Q0_Q1[2]) > value or abs(ecart_H0_H1[0]) > value  or abs(ecart_H0_H1[1]) > value or abs(ecart_H0_H1[2]) > value): # il faut faire un wrapping 2 cylindres
    while Q1_G1_inactive and ecart_length > 0: # il faut faire un wrapping 2 cylindres
     
      Q0, G0 = Q1, G1
      H0, T0 = H1, T1
      segment_length_0 = segment_length_1
      
      g0 = switch_frame(G0, matrix_U)
      g0 = transpose_switch_frame(g0, matrix_V)

      # On fait maintenant le calcul de Q et G, soit v1_U et v2_U
      H1, T1 = find_tangent_points(g0, S_V_cylinder_frame, r_V)
      
      # on veut le H en u
      h0 = switch_frame(H0, matrix_V)
      h0 = transpose_switch_frame(h0, matrix_U)

      # fait un wrapping U 
      Q1, G1 = find_tangent_points(P_U_cylinder_frame, h0, r_U)
      # on regarde si il y a vraiement un wrapping ou pas
      Q1_G1_inactive = determine_if_tangent_points_inactive_single_cylinder(G1, Q1, r_U)
      print("Q1_G1_inactive  = ", Q1_G1_inactive )
      
      segment_length_1 = segment_length_double_cylinder(False, False, P, S, P_U_cylinder_frame, P_V_cylinder_frame, S_U_cylinder_frame, S_V_cylinder_frame, Q1, G1, H1, T1, r_U, r_V, matrix_U, matrix_V)
      ecart_length = segment_length_0 - segment_length_1
      print("ecart_length = ", ecart_length)
    
  else : # faut pas faire, juste avec V
    Q0, G0 = [0,0,0], [0,0,0]
  
  return Q0, G0, H0, T0

#################
def find_tangent_points_iterative_method_5(P, S, P_U_cylinder_frame,P_V_cylinder_frame, S_U_cylinder_frame, S_V_cylinder_frame, r_V, r_U, rotation_matrix_UV, matrix_U, matrix_V) :

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
  Q1_G1_inactive = False
  H1_T1_inactive = False
  ecart_length = 1
  segment_length_1 = 100
  
  while Q1_G1_inactive == False and H1_T1_inactive == False and ecart_length > 0 :
      Q0, G0 = Q1, G1
      H0, T0 = H1, T1
      segment_length_0 = segment_length_1
      
      g0 = switch_frame_UV(H1, matrix_U, matrix_V)
      g0 = transpose_switch_frame(G0, matrix_V)
      H1, T1 = find_tangent_points(g0, S_V_cylinder_frame, r_V)
      H1_T1_inactive = determine_if_tangent_points_inactive_single_cylinder(H1, T1, r_V)
      if H1_T1_inactive : 
        break
      
      h0 = switch_frame_UV(H1, matrix_V, matrix_U)
      Q1, G1 = find_tangent_points(P_U_cylinder_frame, h0, r_U)
      Q1_G1_inactive = determine_if_tangent_points_inactive_single_cylinder(Q1, G1, r_U)
      if Q1_G1_inactive : 
        break
      
      print("Q1_G1_inactive  = ", Q1_G1_inactive )
      print("H1_T1_inactive = ", H1_T1_inactive)
      
      segment_length_1 = segment_length_double_cylinder(False, False, P, S, P_U_cylinder_frame, P_V_cylinder_frame, S_U_cylinder_frame, S_V_cylinder_frame, Q1, G1, H1, T1, r_U, r_V, matrix_U, matrix_V)
      ecart_length = segment_length_0 - segment_length_1
      print("ecart_length = ", ecart_length)
  
  return Q0, G0, H0, T0

def cylinder_V_actif(P_V_cylinder_frame, S_V_cylinder_frame, r_V) : # a voir si vraiment utile
  H, T = find_tangent_points(P_V_cylinder_frame, S_V_cylinder_frame, r_V)
  return determine_if_tangent_points_inactive_single_cylinder(H, T, r_V)


def find_tangent_points_iterative_method_2(P_U_cylinder_frame,P_V_cylinder_frame, S_U_cylinder_frame, S_V_cylinder_frame, r_V, r_U, rotation_matrix_VU, matrix_U, matrix_V) :

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

  # v1_V est notre tangent point H0, v2_V est T, on fait dans le repere V
  Q0, G0 = find_tangent_points(P_U_cylinder_frame, S_U_cylinder_frame, r_U)
  # print("Q0 (premier)= ", Q0)
  # print("G0 (premier)= ", G0)
  ecart_G0_G1 = [1,1,1]

  # while (abs(ecart_G0_G1[0]) > 1e-4  or abs(ecart_G0_G1[1]) > 1e-4  or abs(ecart_G0_G1[2]) > 1e-4) :

  # On passe notre H0 dans le ref du cylindre U --> h0
  # g0 = transpose_switch_frame(G0, rotation_matrix_VU) # ok à priori ...
  # print("g0 avec matrice UV = ", g0)
  g0 = switch_frame(G0, matrix_U)
  g0 = transpose_switch_frame(G0, matrix_V)
  print("g0 avec matrice U et V = ", g0)

  # On fait maintenant le calcul de Q et G, soit v1_U et v2_U
  H0, T0 = find_tangent_points(g0, S_V_cylinder_frame, r_V)
  H0_T0_inactive = determine_if_tangent_points_inactive_single_cylinder(H0, T0, r_V)
  print("H0_T0_inactive = ", H0_T0_inactive)
  
  # Notre G est v1_U, on veut g dans le frame du cylindre V
  # h0 = switch_frame(H0, rotation_matrix_VU)
  # print("h0 avec matrice UV = ", h0)
  # h0 = switch_frame(H0, matrix_V)
  # h0 = transpose_switch_frame(H0, matrix_U)
  # print("h0 avec matrice U et V = ", h0)

  # # On calcule v1_V et v2_V à partir de g0
  # Q1, G1 = find_tangent_points(P_U_cylinder_frame, h0, r_U)

  # ecart_G0_G1 = np.array(G1)-np.array(G0)

  # Q0 = Q1
  # G0 = G1
  
  # g0 = switch_frame(G0, rotation_matrix_VU) # ok à priori ...
  # # print("g0 = ", g0)
  # # print("S_V_cylinder_frame = ", S_V_cylinder_frame)
  # # On fait maintenant le calcul de Q et G, soit v1_U et v2_U
  # H0, T0 = find_tangent_points(g0, S_V_cylinder_frame, r_V)
  
  return Q0, G0, H0, T0



#################

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
   
  #  print("P = ", P)
  #  print("S = ", S)
  #  print("Q = ", Q)
  #  print("G = ", G)
  #  print("H = ", H)
  #  print("T = ", T)

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

       segment_length = norm(Q - np.array(P_U_cylinder_frame)) + Q_G_length + G_H_length + H_T_length_xy + norm(np.array(S_V_cylinder_frame) - T)

   return segment_length