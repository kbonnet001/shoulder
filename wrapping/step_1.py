import numpy as np
from scipy.linalg import norm

# Functions for Step 1
#---------------------
def find_cylinder_frame(center_circle) :

  """Find the frame of the cylinder
  
  INPUT
  - center_circle : 2*array 3*1 coordinates of the first and second circles of the cylinder
  
  OUTPUT
  - cylinder_frame : array 3*3 ortonormal frame for the cylinder"""

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

def find_matrix(cylinder_frame, origin) :
  return np.array([[cylinder_frame[0][0], cylinder_frame[0][1], cylinder_frame[0][2], origin[0]],
                        [cylinder_frame[1][0], cylinder_frame[1][1], cylinder_frame[1][2], origin[1]],
                        [cylinder_frame[2][0], cylinder_frame[2][1], cylinder_frame[2][2], origin[2]],
                        [0, 0, 0, 1]])

def switch_frame(point, matrix) :

  """ Express point in a new frame
  
   INPUT
   - point : array 3*1 coordinates of the point
   - matrix : array 4*4 rotation_matrix and vect
  
   OUTPUT
   - point_new_frame : array 3*1 coordinates of the point in the nex frame
   ----------------------------------
   transformation_matrix = np.array([[rotation_matrix[0][0], rotation_matrix[0][1], rotation_matrix[0][2], vect[0]],
                                     [rotation_matrix[1][0], rotation_matrix[1][1], rotation_matrix[1][2], vect[1]],
                                     [rotation_matrix[2][0], rotation_matrix[2][1], rotation_matrix[2][2], vect[2]],
                                     [0, 0, 0, 1]])
   ----------------------------------"""

  return matrix[0:3, 3] + np.dot(np.transpose(matrix[0:3, 0:3]), point)

def transpose_switch_frame(point, matrix) :

  """Express point in its previous frame
  
  INPUT
  - point : array 3*1 coordinates of the point
  - matrix : array 4*4 rotation_matrix and vect
  
  OUTPUT
  - point_previous_frame : array 3*1 coordinates of the point in its previous frame
  ----------------------------------
  transformation_matrix = np.array([[rotation_matrix[0][0], rotation_matrix[0][1], rotation_matrix[0][2], vect_transition[0]],
                                  [rotation_matrix[1][0], rotation_matrix[1][1], rotation_matrix[1][2], vect_transition[1]],
                                  [rotation_matrix[2][0], rotation_matrix[2][1], rotation_matrix[2][2], vect_transition[2]],
                                  [0, 0, 0, 1]])
  ----------------------------------"""

  vect_transition = np.dot(matrix[0:3, 0:3], [0,0,0] - matrix[0:3, 3])

  return vect_transition + np.dot(matrix[0:3, 0:3], point)