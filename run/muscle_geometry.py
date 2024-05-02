import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import norm # à voir si ok d'utiliser ça

# ---------------
# Datas for plots
# ---------------

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

def data_semi_circle(v1,v2,c,r,num_points=100) :
  # FOR 2D ONLY !
  # v1 and v2 are obstacle via points
  # c is the coordinate of the circle, r the radius
  # default number points = 100

  # Calculation of the normal vect of plan def by v1, v2 and c1
  norm = np.cross(v1 - c, v2 - c)
  norm /= np.linalg.norm(norm)

  # Calculate the angle between v1 and v2
  angle = np.arccos(np.dot((v1 - c) / np.linalg.norm(v1 - c), (v2 - c) / np.linalg.norm(v2 - c)))

  # Calculate points of the semi-circle
  theta = np.linspace(0, angle, num_points)
  semi_circle_points = c + r * np.cos(theta)[:, np.newaxis] * (v1 - c) / np.linalg.norm(v1 - c) + \
                        r * np.sin(theta)[:, np.newaxis] * np.cross(norm, (v1 - c) / np.linalg.norm(v1 - c))

  return semi_circle_points

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

def find_via_points(p0, p1, c, r) :
  # p0 and p1 are the origin and final point
  # c is the center of the circle (2D)/the circulaire obstacle
  # r is the radius of the circulaire obstacle
  v1_x = (p0[0]*r**2 + r*p0[1]*np.sqrt(p0[0]**2+p0[1]**2-r**2))/(p0[0]**2+p0[1]**2) + c[0]
  v1_y = (p0[1]*r**2 - r*p0[0]*np.sqrt(p0[0]**2+p0[1]**2-r**2))/(p0[0]**2+p0[1]**2) + c[1]
  v1_z = 0 # for 2D

  v2_x = (p1[0]*r**2 - r*p1[1]*np.sqrt(p1[0]**2+p1[1]**2-r**2))/(p1[0]**2+p1[1]**2) + c[0]
  v2_y = (p1[1]*r**2 + r*p1[0]*np.sqrt(p1[0]**2+p1[1]**2-r**2))/(p1[0]**2+p1[1]**2) + c[1]
  v2_z = 0 # for 2D

  return [v1_x, v1_y, v1_z], [v2_x, v2_y, v2_z]

def main():

   # ------
   # Inputs
   # ------
   # Points (z=0 for 2D)
   origin_point = [6,1,0]
   final_point =[1,6,0]

   # Points for cylinder (for 2D, choose the same x and same y for both circle)
   center_circle_1 = np.array([5,4,-0.5])
   center_circle_2 = np.array([5,4,0.5])
   center_circle = np.array([(center_circle_1[0]+center_circle_2[0])/2,
    (center_circle_1[1]+center_circle_2[1])/2,(center_circle_1[2]+center_circle_2[2])/2])
   radius = 1

   # ----
   # Plot
   # ----

   # Set figure
   fig = plt.figure()
   ax = fig.add_subplot(111, projection='3d')

   Xc,Yc,Zc = data_cylinder(center_circle_1, center_circle_2, radius )
   ax.plot_surface(Xc, Yc, Zc, alpha=0.5)

   # Via points
   v1, v2 = find_via_points(origin_point, final_point, center_circle_1,radius)

   # Semi-circle between v1 and v2
   semi_circle_points = data_semi_circle(v1,v2,center_circle,radius, 100)
   ax.plot(semi_circle_points[:, 0], semi_circle_points[:, 1], semi_circle_points[:, 2])

   # Set graph
   ax.set_xlabel("X")
   ax.set_ylabel("Y")
   ax.set_zlabel("Z")
   ax.grid(True)

   # Set ax limit
   ax.set_xlim(0,10)
   ax.set_ylim(0,8)
   ax.set_zlim(-0.5,0.5)

   ax.scatter(*origin_point, color='g', label="origin")
   ax.scatter(*final_point, color='b', label="final")
   ax.scatter(*v1, color='r', label="v1")
   ax.scatter(*v2, color='r', label="v2")
   ax.plot(*zip(center_circle_1, center_circle_2), color = 'k')

   ax.plot(*zip(origin_point, final_point), color='b')

   ax.plot(*zip(origin_point, v1), color='g')
   ax.plot(*zip(v1, v2), color='r')
   ax.plot(*zip(v2, final_point), color='b')

   plt.show()

if __name__ == "__main__":
   main()