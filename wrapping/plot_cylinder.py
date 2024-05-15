import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import norm
from wrapping.step_1 import switch_frame, transpose_switch_frame

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