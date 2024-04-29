import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

def data_create_Cylinder(center_x, center_y, radius, height_z) :
  z = np.linspace(0, height_z, 50)
  theta = np.linspace(0, 2*np.pi, 50)
  theta_grid, z_grid=np.meshgrid(theta, z)
  x_grid = radius*np.cos(theta_grid) + center_x
  y_grid = radius*np.sin(theta_grid) + center_y

  return x_grid, y_grid, z_grid

def main():
  
   # Set figure
   fig = plt.figure()
   ax = fig.add_subplot(111, projection='3d')
  
   # Points
   origin_point = [0.1,0.5,0.5]
   final_point=[0.5,0.1,0.5]

   # Points for cylinder
   center_x=0.2
   center_y=0.2
   radius=0.05 # for cylinder "circle"
   height_z=0,1 

   Xc,Yc,Zc = data_create_Cylinder(center_x, center_y, radius, height_z)
   ax.plot_surface(Xc, Yc, Zc, alpha=0.5)
  
   # Set graph
   ax.set_xlabel("X")
   ax.set_ylabel("Y")
   ax.set_zlabel("Z")
   ax.grid(True)
   #plt.plot(origin_point, insertion_point)
   ax.scatter(*origin_point, color='blue', label="origin")
   ax.scatter(*final_point, color='red', label="final")

   ax.plot(*zip(origin_point, final_point), color='green')

   plt.show()

if __name__ == "__main__":
   main()