import numpy as np
from wrapping.step_1 import switch_frame, transpose_switch_frame

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
 