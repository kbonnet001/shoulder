import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import norm
from wrapping.step_1 import switch_frame, transpose_switch_frame

# Functions for Plot
#---------------------------

def create_cylinder(radius, height, num_points = 100) :
  
  """Create a cylinder
  
  INPUT
  - radius : radius of the cylinder
  - height : height of the cylinder
  - num_points : int number of points for representation (default 100)
  
  OUTPUT
  - x_grid, y_grid, z_grid :  array nm_point*num_point coordinates of points for the representation of the cylinder"""
  
  z = np.linspace(-height/2, height/2, num_points)
  theta = np.linspace(0, 2 * np.pi, num_points)
  theta_grid, z_grid = np.meshgrid(theta, z)
  x_grid = radius * np.cos(theta_grid)
  y_grid = radius * np.sin(theta_grid)
  return x_grid, y_grid, z_grid


def apply_transformation(Cylinder, height, num_points = 100):
  """Apply a transformation to the cylinder with a matrix
  
  INPUT
  - matrix : array 4*4 rotation_matrix and vect
  - radius : radius of the cylinder
  - height : height of the cylinder
  - num_points : int number of points for representation (default 100)
  
  OUTPUT
  - x_transformed, y_transformed, z_transformed :  array nm_point*num_point coordinates of points for the 
                                                    representation of the cylinder transformed"""

  x, y, z = create_cylinder(Cylinder.radius, height, num_points)
  shape = x.shape
  ones = np.ones(shape[0] * shape[1])
  points = np.vstack((x.flatten(), y.flatten(), z.flatten(), ones))
  transformed_points = Cylinder.matrix @ points
  x_transformed = transformed_points[0].reshape(shape)
  y_transformed = transformed_points[1].reshape(shape)
  z_transformed = transformed_points[2].reshape(shape)
  
  return x_transformed, y_transformed, z_transformed

def data_cylinder(center_circle_1, center_circle_2, cylinder_frame, radius, num_points = 100) :

  """  Compute datas for plot the cylinder
  The cylinder is charaterized by coordinates of his two circle face and his radius
  
  INPUT
  - center_circle_2 : array 3*1 coordinates of the first circle of the cylinder
  - center_circle_2 : array 3*1 coordinates of the second circle of the cylinder
  - cylinder_frame : array 3*3 local frame of the cylinder
  - radius : radius of the cylinder
  - num_points : int number of points for representation (default 100)
  
  OUTPUT
  - X, Y, Z :  array nm_point*num_point coordinates of points for the representation of the cylinder"""

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

def data_semi_circle(v1, v2, matrix, r, num_points=100) :

  """  Compute datas for plot the semi-circle between bounding fixed tangent points v1 and v2
  
  INPUT
  - v1 : array 3*1 position of the first obstacle tangent point
  - v2 : array 3*1 position of the second obstacle tangent point
  - matrix : array 4*4 rotation_matrix and vect
  - r : radius of the cylinder
  - num_points : int number of points for representation (default 100)
  
  OUTPUT
  - semi_circle_points : array nm_point*n3 coordinates of points for the representation of the semi-circle"""

  # Change frame
  v1 = transpose_switch_frame(v1, matrix)
  v2 = transpose_switch_frame(v2, matrix)
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
    semi_circle_points[i] = switch_frame(semi_circle_points[i], matrix)

  return semi_circle_points

def plot_one_cylinder_obstacle(origin_point, final_point, Cylinder, v1, v2, obstacle_tangent_point_inactive) :

  """   Plot the representation of the single-cylinder obstacle-set algorithm
   
  INPUT
  - origin_point : array 3*1 position of the first point
  - final_point : array 3*1 position of the second point
  - center_circle : 2*array 3*1 coordinates of the first and second circles of the cylinder
  - radius : radius of the cylinder
  - v1 : array 3*1 position of the first obstacle tangent point
  - v2 : array 3*1 position of the second obstacle tangent point
  - obstacle_tangent_point_inactive : bool determine if v1 and v1 or inactive (True) or not (False)
  - matrix : array 4*4 rotation_matrix and vect
  
  OUTPUT
  - None : Plot axis, cylinder, points and muscle path"""

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
  Xc,Yc,Zc = apply_transformation(Cylinder, 0.1, 100)  
  ax.plot_surface(Xc, Yc, Zc, alpha=0.5)

  if (obstacle_tangent_point_inactive == True) : # Muscle path is straight line from origin point to final point
    ax.plot(*zip(origin_point, final_point), color='r')
  else :
    # Semi-circle between v1 and v2
    semi_circle_points = data_semi_circle(v1,v2, Cylinder.matrix,Cylinder.radius, 100)
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
  ax.set_ylim(-0.1,0.1)
  ax.set_zlim(-0.2,0)
  ax.set_xlim(0,0.2)

  plt.title("Single Cylinder Wrapping")
  plt.legend()

  plt.show()

def plot_double_cylinder_obstacle(P, S, Cylinder_U, Cylinder_V, Q, G, H, T, Q_G_inactive, H_T_inactive ) :

  """Plot the representation of the double-cylinder obstacle-set algorithm
  
  INPUT
  - P : array 3*1 position of the first point
  - S : array 3*1 position of the second point
  - Cylinder_U.radius, : radius of the cylinder U
  - radius_V : radius of the cylinder V
  - Q : array 3*1 position of the first obstacle tangent point (in conventional frame)
  - G : array 3*1 position of the second obstacle tangent point (in conventional frame)
  - H : array 3*1 position of the third obstacle tangent point (in conventional frame)
  - T : array 3*1 position of the fourth obstacle tangent point (in conventional frame)
  - matrix_U : array 4*4 rotation_matrix and vect for cylinder U
  - matrix_V : array 4*4 rotation_matrix and vect for cylinder V
  - Q_G_inactive : bool determine if Q and G or inactive (True) or not (False)
  - H_T_inactive : bool determine if H and T or inactive (True) or not (False)
  
  OUTPUT
  - None : Plot axis, cylinder, points and muscle path"""

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
  # Xcu,Ycu,Zcu = data_cylinder(Cylinder_U.c1, Cylinder_U.c2, Cylinder_U.matrix[0:3, 0:3], Cylinder_U.radius)
  Xcu,Ycu,Zcu = apply_transformation(Cylinder_U, 0.1, 100) # ancienne hauteur 0.1
  ax.plot_surface(Xcu, Ycu, Zcu, alpha=0.5)

  # 2nd Cylinder
  # Xcv,Ycv,Zcv = data_cylinder(Cylinder_V.c1, Cylinder_V.c2, Cylinder_V.matrix[0:3, 0:3], Cylinder_V.radius)
  Xcv,Ycv,Zcv = apply_transformation(Cylinder_V, 0.1, 100) # ancienne hauteur 0.1
  ax.plot_surface(Xcv, Ycv, Zcv, alpha=0.5)

  if Q_G_inactive and H_T_inactive: # Muscle path is straight line from origin_point to final_point
    ax.plot(*zip(P, S), color='r')

  elif Q_G_inactive: # single cylinder algorithm with V cylinder
    # Semi-circle between H and T
    semi_circle_points = data_semi_circle(H,T,Cylinder_V.matrix, Cylinder_V.radius, 100)
    ax.plot(semi_circle_points[:, 0], semi_circle_points[:, 1], semi_circle_points[:, 2])

    ax.plot(*zip(P, H), color='g')
    ax.plot(*zip(T, S), color='b')

  elif H_T_inactive: # single cylinder algorithm with U cylinder
    # Semi-circle between Q and G
    semi_circle_points = data_semi_circle(Q,G, Cylinder_U.matrix,Cylinder_U.radius, 100)
    ax.plot(semi_circle_points[:, 0], semi_circle_points[:, 1], semi_circle_points[:, 2])

    ax.plot(*zip(P, Q), color='g')
    ax.plot(*zip(G, S), color='b')

  else: # double cylinder

    # Semi-circle between H and T
    semi_circle_points = data_semi_circle(H,T,Cylinder_V.matrix,Cylinder_V.radius, 100)
    ax.plot(semi_circle_points[:, 0], semi_circle_points[:, 1], semi_circle_points[:, 2])

    # Semi-circle between Q and G
    semi_circle_points = data_semi_circle(Q,G, Cylinder_U.matrix,Cylinder_U.radius, 100)
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
  ax.set_ylim(-0.1,0.1)
  ax.set_zlim(-0.2,0)
  ax.set_xlim(0,0.2)
  
  # ax.set_xlim(-5,5)
  # ax.set_ylim(-5,5)
  # ax.set_zlim(-5,5)

  plt.title("Double Cylinder Wrapping")
  plt.legend()

  plt.show()