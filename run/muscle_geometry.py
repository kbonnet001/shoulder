import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
<<<<<<< HEAD
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
  if (unit_vect == not_unit_vect).all():
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

def find_via_points_xy(p0, p1, r) :

  # Compute xy coordinates of v1 and v2
  #
  # INPUT
  # - p0 : array 3*1 position of the first point
  # - p1 : array 3*1 position of the second point
  # - r : radius of the cylinder * side
  #
  # OUTPUT
  # - v1 = [v1_x, v1_y, 0] : array 3*1 position of the first obstacle via point
  # - v2 = [v2_x, v2_y, 0] : array 3*1 position of the second obstacle via point

=======
from scipy.linalg import norm # à voir si ok d'utiliser ça

# Functions for Step 2
#---------------------

def find_via_points(p0, p1, r) :

  # Compute xy coordinates of v1 and v2
  #
  # INPUT
  # - p0 : array 3*1 position of the first point
  # - p1 : array 3*1 position of the second point
  # - r : radius of the cylinder * side
  #
  # OUTPUT
  # - v1 = [v1_x, v1_y, 0] : array 3*1 position of the first obstacle via point
  # - v2 = [v2_x, v2_y, 0] : array 3*1 position of the second obstacle via point

>>>>>>> 0fa47e04c52e89b607a9a2f5fe966171388f3478
  p0_x2y2 = p0[0] ** 2 + p0[1] ** 2
  if p0_x2y2 == 0:
    raise ValueError("Please choose other coordinates for p0. You mustn't have a bounding-fixed via point with x=y=0.")

  p1_x2y2 = p1[0] ** 2 + p1[1] ** 2
  if p1_x2y2 == 0:
    raise ValueError("Please choose other coordinates for p1. You mustn't have a bounding-fixed via point with x=y=0.")

<<<<<<< HEAD
  if p0[0]**2+p0[1]**2-r**2 < 0 :
    print("You choose p0 in the cylinder. Muscle path is straight line")
    return [0,0,0], [0,0,0]

  elif p1[0]**2+p1[1]**2-r**2 < 0 :
    print("You choose p1 in the cylinder. Muscle path is straight line")
    return [0,0,0], [0,0,0]

  else :
=======
  if p0[0]**2+p0[1]**2-r**2 < 0 : 
    print("You choose p0 in the cylinder. Muscle path is straight line")
    return [0,0,0], [0,0,0]

  elif p1[0]**2+p1[1]**2-r**2 < 0 : 
    print("You choose p1 in the cylinder. Muscle path is straight line")
    return [0,0,0], [0,0,0]

  else : 
>>>>>>> 0fa47e04c52e89b607a9a2f5fe966171388f3478
    v1_x = (p0[0]*r**2 + r*p0[1]*np.sqrt(p0[0]**2+p0[1]**2-r**2))/p0_x2y2 # + c[0]
    v1_y = (p0[1]*r**2 - r*p0[0]*np.sqrt(p0[0]**2+p0[1]**2-r**2))/p0_x2y2 # + c[1]

    v2_x = (p1[0]*r**2 - r*p1[1]*np.sqrt(p1[0]**2+p1[1]**2-r**2))/p1_x2y2 # + c[0]
    v2_y = (p1[1]*r**2 + r*p1[0]*np.sqrt(p1[0]**2+p1[1]**2-r**2))/p1_x2y2 # + c[1]


  return [v1_x, v1_y, 0], [v2_x, v2_y, 0]

def compute_length_v1_v2_xy(v1,v2, r) :

  # Compute xy coordinates of segment lengths in plane
  #
  # INPUT
  # - v1 : array 3*1 position of the first obstacle via point
  # - v2 : array 3*1 position of the second obstacle via point
  # - r : radius of the cylinder * side
  #
  # OUTPUT
  # - ||v1v2||(x,y) : xy coordinates of segment lengths in plane

  if r == 0:
    raise ValueError("Please choose an other radius, positive or negative are accepted. You musn't have r=0")

  # The length of the line segment v1v2(x,y) is found using the law of cosines
  return np.absolute(r*np.arccos(1.0-((v1[0]-v2[0])**2+(v1[1]-v2[1])**2)/(2*r**2)))

def z_coordinates_v1_v2(v1,v2,v1_v2_length_xy, origin_point, final_point) :

  # Compute z coordinates of v1 and v2
  #
  # INPUT
  # - v1 : array 3*1 position of the first obstacle via point
  # - v2 : array 3*1 position of the second obstacle via point
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

<<<<<<< HEAD
def find_via_points(p0, p1, r) :

   # Compute xyz coordinates of v1 and v2
   #
   # INPUT
   # - p0 : array 3*1 position of the first point
   # - p1 : array 3*1 position of the second point
   # - r : radius of the cylinder * side
   #
   # OUTPUT
   # - v1 = [v1_x, v1_y, v1_z] : array 3*1 position of the first obstacle via point
   # - v2 = [v2_x, v2_y, v2_z] : array 3*1 position of the second obstacle via point

   v1, v2 = find_via_points_xy(p0, p1, r)
   v1_v2_length_xy = compute_length_v1_v2_xy(v1, v2, r)
   v1[2], v2[2] = z_coordinates_v1_v2(v1,v2,v1_v2_length_xy, p0, p1)

   return v1, v2

# Functions for Step 3
#----------------------

def determine_if_via_points_inactive_single_cylinder(v1,v2, r) :
=======
# Functions for Step 3 and 4
#---------------------------

def determine_if_via_points_inactive(v1,v2, r) :
>>>>>>> 0fa47e04c52e89b607a9a2f5fe966171388f3478

  # Determine if via points v1 and v2 are inactive
  #
  # /!\ Differences with the B.A. Garner and M.G. Pandy paper !
  #   if Det < 0 : orientation is clockwise
<<<<<<< HEAD
  #   so for a side right-handed (side = 1), we need actived via points
=======
  #   so for a side right-handed (side = 1), we need actived via points 
>>>>>>> 0fa47e04c52e89b607a9a2f5fe966171388f3478
  #   so, if Det*r < 0 ==> determine_if_via_points_inactive = False
  #   (and not "True" as originally presented in the paper)
  #
  # INPUT
  # - v1 : array 3*1 position of the first obstacle via point
  # - v2 : array 3*1 position of the second obstacle via point
  # - r : radius of the cylinder * side
  #
  # OUTPUT
  # - a bool : True if via points are inactive --> Muscle path is straight line from origin point to final point
  #            False if via points are active --> Muscle passes by the two via points

  if (r*(v1[0]*v2[1] - v1[1]*v2[0])<0) :
    return False
  else :
    return True

<<<<<<< HEAD
# Functions for Step 4
#----------------------

def compute_length_v1_v2(v1,v2, v1_v2_length_xy) :

  # Compute length of path segments v1 v2
=======
def compute_length_v1_v2(v1,v2, v1_v2_length_xy) :

  # Compute length of path segments
>>>>>>> 0fa47e04c52e89b607a9a2f5fe966171388f3478
  #
  # INPUT
  # - v1 : array 3*1 position of the first obstacle via point
  # - v2 : array 3*1 position of the second obstacle via point
  # - v1_v2_length_xy : xy coordinates of segment lengths in plane
  #
  # OUTPUT
  # - ||v1v2|| : length of path segments between v1 and v2

  return np.sqrt(v1_v2_length_xy**2+(v2[0]-v1[2])**2)

<<<<<<< HEAD
def segment_length_single_cylinder(obstacle_via_point_inactive, origin_point, final_point, v1, v2, r) :

  # Compute length of path segments
  #
  # INPUT
  # - obstacle_via_point_inactive : bool determine if v1 and v1 or inactive (True) or not (False)
  # - origin_point : array 3*1 position of the first point
  # - final_point : array 3*1 position of the second point
  # - v1 : array 3*1 position of the first obstacle via point
  # - v2 : array 3*1 position of the second obstacle via point
  # - r : radius of the cylinder * side
  #
  # OUTPUT
  # - segment_length : length of path segments

  if (obstacle_via_point_inactive == True) : # Muscle path is straight line from origin_point to final_point
   segment_length = norm(np.array(final_point)-np.array(origin_point))

  else :
   v1_v2_length_xy = compute_length_v1_v2_xy(v1, v2, r)
   v1_v2_length = compute_length_v1_v2(v1,v2, v1_v2_length_xy)
   segment_length = norm (v1 - np.array(origin_point)) + v1_v2_length + norm(np.array(final_point) - v2)

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
   # - v1o : array 3*1 position of the first obstacle via point (in conventionnal frame)
   # - v2o : array 3*1 position of the second obstacle via point (in conventionnal frame)
   # - obstacle_via_point_inactive : bool determine if v1 and v1 or inactive (True) or not (False)
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
   # Via points
   v1, v2 = find_via_points(P_cylinder_frame, S_cylinder_frame, r)
   v1_v2_length_xy = compute_length_v1_v2_xy(v1, v2, r)

   # ------
   # Step 3
   # ------
   obstacle_via_point_inactive = determine_if_via_points_inactive_single_cylinder(v1,v2, r)

   # ------
   # Step 4
   # ------
   segment_length = segment_length_single_cylinder(obstacle_via_point_inactive, P_cylinder_frame, S_cylinder_frame, v1, v2, r)

   # ------
   # Step 5
   # ------
   v1o = switch_frame(v1, np.transpose(cylinder_frame), cylinder_origin)
   v2o = switch_frame(v2, np.transpose(cylinder_frame), cylinder_origin)

   return v1o, v2o, obstacle_via_point_inactive, segment_length

=======
>>>>>>> 0fa47e04c52e89b607a9a2f5fe966171388f3478
# Functions for Plot
#---------------------------

def data_cylinder(center_circle_1, center_circle_2, radius, num_points = 100) :

  # Compute datas for plot the cylinder
  # The cylinder is charaterized by coordinates of his two circle face and his radius
  #
  # INPUT
  # - center_circle_2 : array 3*1 coordinates of the first circle of the cylinder
  # - center_circle_2 : array 3*1 coordinates of the second circle of the cylinder
  # - radius : radius of the cylinder
  # - num_points : int number of points for representation (default 100)
  #
  # OUTPUT
  # - X, Y, Z :  array nm_point*num_point coordinates of points for the representation of the cylinder

  # Create a unit vector in direction of axis
  v_cylinder=center_circle_2-center_circle_1
  v_unit=(v_cylinder)/norm(v_cylinder)

  # Make some vector not in the same direction as v
  not_v_unit = np.array([1, 0, 0])
  if (v_unit == not_v_unit).all():
    not_v_unit = np.array([0, 1, 0])

  # Make a normalized vector perpendicular to v
  n1 = np.cross(v_unit, not_v_unit)/norm(np.cross(v_unit, not_v_unit))

  # Make unit vector perpendicular to v and n1
  n2 = np.cross(v_unit, n1)

  # Surface ranges over t from 0 to length of axis and 0 to 2*pi
<<<<<<< HEAD
  t = np.linspace(0, norm(v_cylinder), num_points)
  theta = np.linspace(0, 2 * np.pi, num_points)
=======
  t = np.linspace(0, norm(v_cylinder), 100)
  theta = np.linspace(0, 2 * np.pi, 100)
>>>>>>> 0fa47e04c52e89b607a9a2f5fe966171388f3478
  t, theta = np.meshgrid(t, theta)

  # Generate coordinates for surface
  X, Y, Z = [center_circle_1[i] + v_unit[i] * t + radius * np.sin(theta) * n1[i] + radius * np.cos(theta) * n2[i] for i in [0, 1, 2]]
  return X, Y, Z

<<<<<<< HEAD
def data_semi_circle(v1, v2, cylinder_origin, cylinder_frame, r, num_points=100) :
=======
def data_semi_circle(v1, v2, c, r, num_points=100) :
>>>>>>> 0fa47e04c52e89b607a9a2f5fe966171388f3478

  # Compute datas for plot the semi-circle between bounding fixed via points v1 and v2
  #
  # INPUT
  # - v1 : array 3*1 position of the first obstacle via point
  # - v2 : array 3*1 position of the second obstacle via point
<<<<<<< HEAD
  # - cylinder_origin : array 3*1 coordinates of the center of the cylinder
  # - cylinder_frame : array 3*3 local frame of the cylinder
  # - r : radius of the cylinder
=======
  # - c : array 3*1 position of the center of the circle passing through v1 and v2
>>>>>>> 0fa47e04c52e89b607a9a2f5fe966171388f3478
  # - num_points : int number of points for representation (default 100)
  #
  # OUTPUT
  # - semi_circle_points : array nm_point*n3 coordinates of points for the representation of the semi-circle

<<<<<<< HEAD
  # Change frame
  v1 = transpose_switch_frame(v1, cylinder_frame, [0,0,0] - cylinder_origin)
  v2 = transpose_switch_frame(v2, cylinder_frame, [0,0,0] - cylinder_origin)
  c = np.array([0,0, (v1[2]+v2[2])/2])

=======
>>>>>>> 0fa47e04c52e89b607a9a2f5fe966171388f3478
  # Calculation of the normal vect of plan def by v1, v2 and c1
  norm = np.cross(v1 - c, v2 - c)
  norm /= np.linalg.norm(norm)

  # Calculate the angle between v1 and v2
  angle = np.arccos(np.dot((v1 - c) / np.linalg.norm(v1 - c), (v2 - c) / np.linalg.norm(v2 - c)))

  # Calculate points of the semi-circle
  theta = np.linspace(0, angle, num_points)
  semi_circle_points = c + r * np.cos(theta)[:, np.newaxis] * (v1 - c) / np.linalg.norm(v1 - c) + \
                        r * np.sin(theta)[:, np.newaxis] * np.cross(norm, (v1 - c) / np.linalg.norm(v1 - c))

<<<<<<< HEAD
  for i in range (len(semi_circle_points)) :
    semi_circle_points[i] = switch_frame(semi_circle_points[i], np.transpose(cylinder_frame), cylinder_origin)

  return semi_circle_points

def plot_one_cylinder_obstacle(origin_point, final_point, center_circle, radius, v1, v2, obstacle_via_point_inactive, segment_length, cylinder_origin, cylinder_frame) :
=======
  return semi_circle_points

def plot_one_cylinder_obstacle(origin_point, final_point, center_circle_1, center_circle_2, radius, v1, v2, obstacle_via_point_inactive, segment_length) :
>>>>>>> 0fa47e04c52e89b607a9a2f5fe966171388f3478

   # Plot the representation of the single-cylinder obstacle-set algorithm
   #
   # INPUT
   # - origin_point : array 3*1 position of the first point
   # - final_point : array 3*1 position of the second point
<<<<<<< HEAD
   # - center_circle : 2*array 3*1 coordinates of the first and second circles of the cylinder
   # - radius : radius of the cylinder
   # - v1 : array 3*1 position of the first obstacle via point
   # - v2 : array 3*1 position of the second obstacle via point
   # - obstacle_via_point_inactive : bool determine if v1 and v1 or inactive (True) or not (False)
   # - segment_length : length of path segments
   # - cylinder_origin : array 3*1 coordinates of the center of the cylinder
   # - cylinder_frame : array 3*3 local frame of the cylinder
=======
   # - center_circle_2 : array 3*1 coordinates of the first circle of the cylinder
   # - center_circle_2 : array 3*1 coordinates of the second circle of the cylinder
   # - radius : radius of the cylinder
   # - obstacle_via_point_inactive : bool determine if v1 and v1 or inactive (True) or not (False)
   # - segment_length : length of path segments
>>>>>>> 0fa47e04c52e89b607a9a2f5fe966171388f3478
   #
   # OUTPUT
   # - None : Plot axis, cylinder, points and muscle path

   # Set figure
   fig = plt.figure()
   ax = fig.add_subplot(111, projection='3d')

   # Bouding-fixed via point
   ax.scatter(*origin_point, color='g', label="Origin point")
   ax.scatter(*final_point, color='b', label="Final point")

   #Obstacle via points
   ax.scatter(*v1, color='r', label="v1")
   ax.scatter(*v2, color='r', label="v2")

   # Cylinder
<<<<<<< HEAD
   Xc,Yc,Zc = data_cylinder(center_circle[0], center_circle[1], radius )
   ax.plot_surface(Xc, Yc, Zc, alpha=0.5)
   ax.plot(*zip(center_circle[0], center_circle[1]), color = 'k')
=======
   Xc,Yc,Zc = data_cylinder(center_circle_1, center_circle_2, radius )
   ax.plot_surface(Xc, Yc, Zc, alpha=0.5)
   ax.plot(*zip(center_circle_1, center_circle_2), color = 'k')
>>>>>>> 0fa47e04c52e89b607a9a2f5fe966171388f3478

   if (obstacle_via_point_inactive == True) : # Muscle path is straight line from origin point to final point
    ax.plot(*zip(origin_point, final_point), color='r')
   else :
    # Semi-circle between v1 and v2
<<<<<<< HEAD
    semi_circle_points = data_semi_circle(v1,v2,cylinder_origin, cylinder_frame,radius, 100)
=======
    center_circle=np.array([center_circle_1[0], center_circle_1[1], (v1[2]+v2[2])/2])
    semi_circle_points = data_semi_circle(v1,v2,center_circle,radius, 100)
>>>>>>> 0fa47e04c52e89b607a9a2f5fe966171388f3478
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

<<<<<<< HEAD
   plt.title("Single Cylinder Wrapping")
=======
   plt.title("Cylinder Wrapping")
>>>>>>> 0fa47e04c52e89b607a9a2f5fe966171388f3478
   plt.legend()
  #  plt.legend(["obstacle_via_point_inactive : "+ ("True" if obstacle_via_point_inactive else "False")])
  #  plt.legend(["segment_length : ", segment_length])
   plt.show()
<<<<<<< HEAD

def main():
=======

# Algorithm
#---------------------------

def single_cylinder_obstacle_set_algorithm(origin_point, final_point, radius, side) :

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
   #
   # OUTPUT
   # - v1 : array 3*1 position of the first obstacle via point
   # - v2 : array 3*1 position of the second obstacle via point
   # - obstacle_via_point_inactive : bool determine if v1 and v1 or inactive (True) or not (False)
   # - segment_lenght : length of path segments

   # ------
   # Step 1
   # ------
   r = radius * side

   # ------
   # Step 2
   # ------
   # Via points
   v1, v2 = find_via_points(origin_point, final_point,r)
   v1_v2_length_xy = compute_length_v1_v2_xy(v1, v2, r)
   v1[2], v2[2] = z_coordinates_v1_v2(v1,v2,v1_v2_length_xy, origin_point, final_point)

   # --------------------------
   # Step 3 - v1, v2 inactive ?
   # --------------------------
   obstacle_via_point_inactive = determine_if_via_points_inactive(v1,v2, r)

   if (obstacle_via_point_inactive == True) : # Muscle path is straight line from origin_point to final_point
    segment_lenght = norm(np.array(final_point)-np.array(origin_point))

   else :
    # ------
    # Step 4
    # ------
    v1_v2_length = compute_length_v1_v2(v1,v2, v1_v2_length_xy)

    segment_lenght = norm (v1 - np.array(origin_point)) + v1_v2_length + norm(np.array(final_point) - v2)

   return v1, v2, obstacle_via_point_inactive, segment_lenght

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
   origin_point = [-5, -1,2]
   final_point =[-1,-5,-1]

   # Points for cylinder
   center_circle_1 = np.array([0,0,-4])
   center_circle_2 = np.array([0,0,4])
   radius = 1
   side = 1 # Choose 1 for the right-handed side, -1 for the left-handed side (with respect to the z-axis)

   #
   v1, v2, obstacle_via_point_inactive, segment_length = single_cylinder_obstacle_set_algorithm(origin_point, final_point, radius, side)

   # ----
   # Plot
   # ----
   plot_one_cylinder_obstacle(origin_point, final_point, center_circle_1, center_circle_2, radius, v1, v2, obstacle_via_point_inactive, segment_length)

   print("obstacle_via_point_inactive = ",obstacle_via_point_inactive)
   print("segment_length = ", round(segment_length, 2))
>>>>>>> 0fa47e04c52e89b607a9a2f5fe966171388f3478

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
   origin_point = [-5, -1,2]
   final_point =[-1,-5,-1]

   # Points for cylinder
   center_circle = [np.array([0,0,-4]),np.array([0,2,4])]
   radius = 1
   side = 1 # Choose 1 for the right-handed side, -1 for the left-handed side (with respect to the z-axis)

   show_details = True

   # Other inputs -----------------------------------------
   cylinder_origin = (center_circle[1] + center_circle[0]) / 2
   cylinder_frame = find_cylinder_frame(center_circle)

   # --------------------------------------
   # Single cylinder obstacle set algorithm
   # --------------------------------------
   v1, v2, obstacle_via_point_inactive, segment_length = single_cylinder_obstacle_set_algorithm(origin_point, final_point, radius, side, cylinder_origin, cylinder_frame)

   # ----
   # Plot
   # ----
   plot_one_cylinder_obstacle(origin_point, final_point, center_circle, radius, v1, v2, obstacle_via_point_inactive, segment_length, cylinder_origin, cylinder_frame)

   if (show_details) :
    print("origin_point = ", origin_point)
    print("final_point = ", final_point)
    print("v1 = ", v1)
    print("v2 = ", v2)
    print("obstacle_via_point_inactive = ",obstacle_via_point_inactive)
    print("segment_length = ", round(segment_length, 2))

if __name__ == "__main__":
<<<<<<< HEAD
   main()
=======
   main()

>>>>>>> 0fa47e04c52e89b607a9a2f5fe966171388f3478
