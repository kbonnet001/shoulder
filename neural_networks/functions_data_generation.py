import numpy as np
from wrapping.algorithm import single_cylinder_obstacle_set_algorithm, double_cylinder_obstacle_set_algorithm
from wrapping.plot_cylinder import plot_one_cylinder_obstacle, plot_double_cylinder_obstacle
from scipy.linalg import norm
from neural_networks.discontinuities import *

def ensure_unique_column_names(dofs_list) : 
   """
    Ensure unique column names by appending a suffix to duplicates.

    This function takes a list of names and appends a numeric suffix
    to duplicate names to ensure all names in the list are unique.

    Args:
   - dofs_list (list of str): A list of column names that may contain duplicates.

    Returns:
    - list of str: A list of unique column names.
    """
    
   name_counts = {}
   unique_names = []
   
   # Iterate over each name in the input list
   for name in dofs_list:
      if name in name_counts:
         # Increment the count and create a unique name with a suffix
         name_counts[name] += 1
         unique_name = f"{name}_{name_counts[name]}"
      else:
         # Initialize the count and use the original name
         name_counts[name] = 0
         unique_name = name
         
      # Add the unique name to the list
      unique_names.append(unique_name)

   return unique_names

def compute_q_ranges(model):
   """
   Get ranges of each q and names.

   This function extracts the ranges of joint angles (q) and their corresponding names
   from a given biorbd model.

   Args:
      model (biorbd.Model): The biorbd model containing segments and degrees of freedom.

   Returns:
      tuple: A tuple containing:
         - q_ranges (list of list): A list of lists where each sublist contains the min
            and max values of a joint angle range.
         - q_ranges_names_with_dofs (list of str): A list of strings representing the
            names of each degree of freedom, including the segment name.
   """
   # Get the names of each segment in the model (2, 4, 6, 7)
   segment_names = [model.segment(i).name().to_string() for i in range(model.nbSegment())]
   
   q_ranges = []
   q_ranges_names_with_dofs = []
   
   for segment_index in range(model.nbSegment()) :
      # Get the range of motion for each degree of freedom in the segment
      q_ranges_segment = [[ranges.min(), ranges.max()] for ranges in model.segment(segment_index).QRanges()]
      q_ranges+= q_ranges_segment
      
      # Get the names of the degrees of freedom for the segment
      dof_names = [model.segment(segment_index).nameDof(i).to_string() for i in range(model.segment(segment_index).nbQ())]
      q_ranges_names_with_dofs += [f"{segment_names[segment_index]}_{dof}" for dof in dof_names]
      
   # Ensure all degree of freedom names are unique
   q_ranges_names_with_dofs = ensure_unique_column_names(q_ranges_names_with_dofs)
   
   return q_ranges, q_ranges_names_with_dofs

def initialisation_generation(model, muscle_index, cylinders) :
   """
   Initialization before generation.

   This function initializes the required components before generating a specific
   configuration and datas for a given muscle in the model. It prepares the cylinders and
   muscle points for the simulation.

   Args:
      model (biorbd.Model): The biorbd model containing segments and muscles.
      muscle_index (int): The index of the selected muscle to be initialized.
      cylinders (list of Cylinder): A list of cylinders associated with the selected muscle.
   """
   # Get the names of each segment in the model
   segment_names = [model.segment(i).name().to_string() for i in range(model.nbSegment())]

   # Set the initial position of all degrees of freedom to 0.0
   q_initial = np.array([0.0 for i in range(model.nbQ())])
   
   # Compute the initial segment index and global coordinate system for each cylinder
   for cylinder in cylinders : 
      cylinder.compute_seg_index_and_gcs_seg_0(q_initial, model, segment_names)
      
   # Update the positions of the origin and insertion points of the muscle
   origin_muscle, insertion_muscle = update_points_position(model, [0, -1], muscle_index, q_initial) # global initial
   points = [origin_muscle, insertion_muscle]
   
   for k in range(len(cylinders)) : 
      # Warning: This function only works for PECM2 and PECM3 (2 cylinders)
      # More information in the "compute_new_radius" description
      cylinders[k].compute_new_radius(points[k])


def update_points_position(model, point_index, muscle_index, q) : 
   """
   Update coordinates of specified muscle points.

   This function updates the coordinates of specified points on a muscle within the biorbd model
   based on the given generalized coordinates (q).

   Args:
      model (biorbd.Model): The biorbd model containing segments and muscles.
      point_index (list of int): List of indices for the muscle points whose coordinates are desired.
      muscle_index (int): The index of the selected muscle.
      q (array): An array of generalized coordinates.

   Returns:
      Generator: A generator yielding the coordinates of the specified muscle points.
   """
   # Update
   model.updateMuscles(q) 
   mus = model.muscle(muscle_index) 
   model.UpdateKinematicsCustom(q)

   # Find coordinates of the specified points (e.g., origin and insertion points) in the global frame
   points = []
   for k in (point_index) : 
      points.append(mus.musclesPointsInGlobal(model, q)[k].to_array()) 
   
   return (points[k] for k in range (len(points)))

def find_index_muscle(model, muscle):
   
   """ 
   Find the index of the selected muscle by its name in the biorbd model.

   Args:
      model (biorbd.Model): The biorbd model containing muscles.
      muscle (str): Name of the muscle whose index is to be found.

   Returns:
      int: The position of the muscle in the list.

   Raises:
      ValueError: If the muscle name is not found in the model.
   """

   # Extract muscle names from the model
   muscle_names = [model.muscle(i).name().to_string() for i in range(model.nbMuscles())]

   try:
      # Find the index of the given muscle name
      position = muscle_names.index(muscle)
      return position
   except ValueError:
      # If muscle name is invalid, raise an error with a list of valid names
      valid_muscles = ", ".join(muscle_names)
      raise ValueError(f"Invalid muscle name '{muscle}'. You must choose a valid muscle from this list: [{valid_muscles}]")

def compute_segment_length(model, cylinders, q, origin_muscle, insertion_muscle, plot = False, plot_cadran = False) :

   """Compute segment length
   
   Args:
   - model: biorbd model
   - cylinders: List of muscle's cylinders (0, 1, or 2 cylinders)
   - q: array, randomly generated joint angles
   - origin_muscle: array 1*3, coordinates of the muscle origin
   - insertion_muscle: array 1*3, coordinates of the muscle insertion
   - plot: bool, (default = False) whether to plot cylinder(s), points, and muscle path
   
   Returns:
   - segment_length: length of the muscle path
   - data_ignored: information about ignored data during computation because of wrapping errors
   """
   
   # Create a rotation matrix to align axes (the model uses y-axis as up instead of z-axis)
   # This adjustment is necessary for both single and double cylinder algorithms
   matrix_rot_zy = np.array([[0,0,1,0], [1,0,0,0], [0,1,0,0], [0,0,0,1]])

   # Apply rotation to the origin and insertion muscle points
   origin_muscle_rot = np.dot(matrix_rot_zy[0:3, 0:3], origin_muscle)
   insertion_muscle_rot = np.dot(matrix_rot_zy[0:3, 0:3], insertion_muscle)
   points = [origin_muscle_rot, insertion_muscle_rot]
   
   for cylinder in cylinders :
      # Update the segment matrix based on the model and joint angles
      cylinder.compute_new_matrix_segment(model, q) 
      # Apply the rotation matrix to align cylinder orientation
      cylinder.compute_matrix_rotation_zy(matrix_rot_zy) 
   
   # Compute the new radius for each cylinder based on the rotated points
   for k in range(len(cylinders)) : 
      cylinders[k].compute_new_radius(points[k])
   
   if (len(cylinders) == 0) :
      # If there are no cylinders, compute the muscle path as a straight line
      # Warning: This is a simple straight line and does not consider any via points
      segment_length = norm(np.array(origin_muscle_rot) - np.array(insertion_muscle_rot))
      points_not_in_cylinder = []
      bool_inactive = []

   elif (len(cylinders) == 1) :
      # Compute the muscle path when there is a single cylinder
      Q_rot, T_rot, Q_T_inactive, segment_length = single_cylinder_obstacle_set_algorithm(origin_muscle_rot, insertion_muscle_rot, cylinders[0], plot_cadran)
      points_not_in_cylinder = [origin_muscle_rot, insertion_muscle_rot]
      bool_inactive = [Q_T_inactive]
      
      if plot == True : 
         plot_one_cylinder_obstacle(origin_muscle_rot, insertion_muscle_rot, cylinders[0], Q_rot, T_rot, Q_T_inactive)

   else : # (len(cylinders) == 2 ) 
      # Compute the muscle path when there are two cylinders
      Q_rot, G_rot, H_rot, T_rot, Q_G_inactive, H_T_inactive , segment_length  = double_cylinder_obstacle_set_algorithm(origin_muscle_rot, insertion_muscle_rot, cylinders[0], cylinders[1], plot_cadran)
      points_not_in_cylinder = [origin_muscle_rot, H_rot, G_rot, insertion_muscle_rot]
      bool_inactive = [Q_G_inactive, H_T_inactive]
      
      if plot == True : 
         plot_double_cylinder_obstacle(origin_muscle_rot, insertion_muscle_rot, cylinders[0], cylinders[1], Q_rot, G_rot, H_rot, T_rot, Q_G_inactive, H_T_inactive)
   
   # Check for any ignored data points during computation
   data_ignored = find_if_data_ignored(cylinders, points_not_in_cylinder, bool_inactive)
      
   return segment_length, data_ignored
