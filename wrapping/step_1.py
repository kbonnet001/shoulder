import numpy as np

# Functions for Step 1
#---------------------

def switch_frame(points, matrix) :

  """ Express point in a new frame/ global frame
  
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

  # return (matrix @ np.concatenate((point, [1])))[:3]
  if isinstance(points[0], (list, np.ndarray)):  # Check if points is a list of points
    return [(matrix @ np.concatenate((point, [1])))[:3] for point in points]
  else:  # Single point
      return (matrix @ np.concatenate((points, [1])))[:3]

def transpose_switch_frame(points, matrix) :

  """Express point in its previous frame/ in local frame
  
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

  # return (np.linalg.inv(matrix)@ np.concatenate((point, [1])))[:3]
  if isinstance(points[0], (list, np.ndarray)):  # Check if points is a list of points
    return [(np.linalg.inv(matrix)@ np.concatenate((point, [1])))[:3] for point in points]
  else:  # Single point
      return (np.linalg.inv(matrix)@ np.concatenate((points, [1])))[:3]

def switch_frame_UV(points, matrix_U, matrix_V) : 
  """Express point, its local frame, to an other local frame
  
  INPUT
  - point : array 3*1 coordinates of the point
  - matrix_U : array 4*4 rotation_matrix and vect, local frame of point
  - matrix_V : array 4*4 rotation_matrix and vect, new local frame 
  
  OUTPUT
  - point_new_local_frame : array 3*1 coordinates of the point in the new local frame"""
  
  points = switch_frame(points, matrix_U)
  points = transpose_switch_frame(points, matrix_V)
  return points