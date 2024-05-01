import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import norm # à voir si ok d'utiliser ça

def findEquation(point_1,point_2):
  # Point_1 and point_2 are two points (origin and final)
  # point = [point_x, point_y, point_z]
  # z=ax+by+c

  # Calculation of the leading coefficients a and b
  a = (point_2[2] - point_1[2]) / (point_2[0] - point_1[0])
  b = (point_2[2] - point_1[2]) / (point_2[1] - point_1[1])

  # Calculation of the intercept c
  c = point_1[2] - a * point_1[0] - b * point_1[1]

  return a, b, c

def data_cylinder(center_circle_1, center_circle_2, radius) :
  # Center_circle_1 and 2 are two points caracterizing the position of the cylinder

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
  t = np.linspace(0, norm(v_cylinder), 100)
  theta = np.linspace(0, 2 * np.pi, 100)
  t, theta = np.meshgrid(t, theta)

  # Generate coordinates for surface
  X, Y, Z = [center_circle_1[i] + v_unit[i] * t + radius * np.sin(theta) * n1[i] + radius * np.cos(theta) * n2[i] for i in [0, 1, 2]]
  return X, Y, Z

def main():

   # Set figure
   fig = plt.figure()
   ax = fig.add_subplot(111, projection='3d')

   # Points
   origin_point = [0.1,0.5,0.5]
   final_point =[0.5,0.1,0.1]

   # Points for cylinder
   center_circle_1 = np.array([0.2,0.2,0.3])
   center_circle_2 = np.array([0.5,0.5,0.5])
   radius = 0.05

   Xc,Yc,Zc = data_cylinder(center_circle_1, center_circle_2, radius )
   ax.plot_surface(Xc, Yc, Zc, alpha=0.5)

   # Set graph
   ax.set_xlabel("X")
   ax.set_ylabel("Y")
   ax.set_zlabel("Z")
   ax.grid(True)

   # Set ax limit
   #ax.set_xlim(0,1)
   #ax.set_ylim(0,1)
   #ax.set_zlim(0,1)

   ax.scatter(*origin_point, color='blue', label="origin")
   ax.scatter(*final_point, color='red', label="final")
   ax.plot(*zip(center_circle_1, center_circle_2), color = 'k')

   ax.plot(*zip(origin_point, final_point), color='green')

   plt.show()

if __name__ == "__main__":
   main()