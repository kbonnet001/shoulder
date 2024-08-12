import numpy as np

# Functions for Step 1
#---------------------

def switch_frame(points, matrix) :

  """ Transforms a point from local to global frame.
  
   Args
  - point(s): 3x1 array (3xn array), coordinates of the point.
  - matrix: 4x4 array, rotation matrix and translation vector.
  
   Returns
   - point(s)_global_frame: 3x1 array (3xn array), coordinates of the point in global frame.
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

  """ Transforms a point from global to local frame.

  Args:
  - point(s): 3x1 array (3xn array), coordinates of the point.
  - matrix: 4x4 array, rotation matrix and translation vector.

  Returns:
  - point(s)_local_frame: 3x1 array (3xn array), coordinates of the point in local frame.
  ----------------------------------
  transformation_matrix = np.array([[rotation_matrix[0][0], rotation_matrix[0][1], rotation_matrix[0][2], vect_transition[0]],
                                    [rotation_matrix[1][0], rotation_matrix[1][1], rotation_matrix[1][2], vect_transition[1]],
                                    [rotation_matrix[2][0], rotation_matrix[2][1], rotation_matrix[2][2], vect_transition[2]],
                                    [0, 0, 0, 1]])
  ----------------------------------
  """

  # return (np.linalg.inv(matrix)@ np.concatenate((point, [1])))[:3]
  if isinstance(points[0], (list, np.ndarray)):  # Check if points is a list of points
    return [(np.linalg.inv(matrix)@ np.concatenate((point, [1])))[:3] for point in points]
  else:  # Single point
      return (np.linalg.inv(matrix)@ np.concatenate((points, [1])))[:3]

def switch_frame_UV(point, matrix_U, matrix_V) : 
  """ Expresses a point in its local frame U as coordinates in another local frame V.

  Args:
  - point: 3x1 array, coordinates of the point.
  - matrix_U: 4x4 array, rotation matrix and translation vector of the local frame U.
  - matrix_V: 4x4 array, rotation matrix and translation vector of the new local frame V.

  Returns:
  - point_new_local_frame: 3x1 array, coordinates of the point in the new local frame V.
  """
  
  # points local U --> points global
  point = switch_frame(point, matrix_U)
  # points global --> points local V
  point = transpose_switch_frame(point, matrix_V)
  return point