import numpy as np
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
  if (unit_vect == [1,0,0]).all() or (unit_vect == [-1,0,0]).all():
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