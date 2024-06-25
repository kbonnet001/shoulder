import numpy as np


# Functions for Step 1
#---------------------

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

  return (matrix @ np.concatenate((point, [1])))[:3]

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


  # point = np.array(([1, 2, 3, 1], [1, 2, 3, 1],[1, 2, 3, 1], [1, 2, 3, 1], [1, 2, 3, 1], [1, 2, 3, 1], [1, 2, 3, 1])).T
  rot = matrix[:3, :3].T  
  rototrans = np.eye(4)
  rototrans[:3, :3] = rot
  rototrans[:3, 3] = -rot @ matrix[:3, 3]
  # rototrans @ point
  return (rototrans @ np.concatenate((point, [1])))[:3]


def switch_frame_UV(point, matrix_U, matrix_V) : 
  point = switch_frame(point, matrix_U)
  point = transpose_switch_frame(point, matrix_V)
  return point
