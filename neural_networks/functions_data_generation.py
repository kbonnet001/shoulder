import numpy as np
from wrapping.algorithm import single_cylinder_obstacle_set_algorithm, double_cylinder_obstacle_set_algorithm
from wrapping.plot_cylinder import plot_one_cylinder_obstacle, plot_double_cylinder_obstacle
from scipy.linalg import norm
from openpyxl import load_workbook
from neural_networks.discontinuities import *

def compute_q_ranges_segment(model, segment_selected) : 
    # segment_names
    # ['thorax', 'spine', 'clavicle_effector_right', 'clavicle_right', 'scapula_effector_right', 'scapula_right', 
    # 'humerus_right', 'ulna_effector_right', 'ulna_right', 'radius_effector_right', 'radius_right', 'hand_right']
    
    segment_names = [model.segment(i).name().to_string() for i in range(model.nbSegment())]
    segment_index = segment_names.index(segment_selected) 

    # humerus_dof_names = [model.segment(humerus_index).nameDof(i).to_string() for i in 
    #                     range(model.segment(humerus_index).nbQ())]
    
    q_ranges = [[ranges.min(), ranges.max()] for ranges in model.segment(segment_index).QRanges()]
    return q_ranges

def initialisation_generation(model, muscle_selected, cylinders) :
   # Find index of the muscle selected
   muscle_names = [model.muscle(i).name().to_string() for i in range(model.nbMuscles())]
   muscle_index = find_index_muscle(muscle_selected, muscle_names)

   # Find name of segments and matrix of cylinders  
   segment_names = [model.segment(i).name().to_string() for i in range(model.nbSegment())]
   # q_initial = np.array([0.,-0.01,0.,0.05])
   q_initial = np.array([0.0, 0.0, 0.0, 0.0]) 
   for cylinder in cylinders : 
      cylinder.compute_seg_index_and_gcs_seg_0(q_initial, model, segment_names)
   
   origin_muscle, insertion_muscle = update_points_position(model, muscle_index, q_initial) # global initial
   points = [origin_muscle, insertion_muscle]
   for k in range(2) : 
      cylinders[k].compute_new_radius(points[k])
   
   return muscle_index

def update_points_position(model, muscle_index, q) : 
   # Updates
   model.updateMuscles(q) 
   mus = model.muscle(muscle_index) 
   
   model.UpdateKinematicsCustom(q)

   # Find coordinates of origin point (P) and insertion point (S)
   origin_muscle = mus.musclesPointsInGlobal(model, q)[0].to_array() 
   insertion_muscle = mus.musclesPointsInGlobal(model, q)[-1].to_array() 
   
   # ces points sont dans le repere global (verifiee)
   # cela revient à faire :
   # switch_frame([0.016, -0.0354957, 0.005], gcs_seg))
   
   return origin_muscle, insertion_muscle
       

def find_index_muscle(muscle, muscle_names):
   
   """ Find index of the muscle selected in the name list
   
   INPUT : 
   - muscle : string, name of the muscle selected
   - muscle_names : string, list of muscle names
   
   OUTPUT : 
   - position : int, position of the muscle in the list"""

   try:
      position = muscle_names.index(muscle) 
      return position
   except ValueError:
      valid_muscles = ", ".join(muscle_names)
      raise ValueError(f"Invalid muscle name '{muscle}'. You must choose a valid muscle from this list: [{valid_muscles}]")

def compute_segment_length(model, cylinders, q, origin_muscle, insertion_muscle, plot = False, plot_cadran = False) :

   """Compute segment length
   
   INPUT : 
   - model : model
   - cylinders : List of muscle's cylinder (0, 1 or 2 cylinders)
   - q : array 4*2, q randomly generated
   - origin_muscle : array 1*3, coordinate of the origin point
   - insertion_muscle : array 1*3, coordinate of the insertion point
   - segment_U_index : int, index of the segment of U cylinder (if it exist)
   - segment_V_index : int, index of the segment of V cylinder (if it exist)
   - gcs_seg_U_0 : gcs 0 (inital) of the segment U (if it exist)
   - gcs_seg_V_0 : gcs 0 (inital) of the segment V (if it exist)
   - plot = bool, (default = False) plot cylinder(s), points and muscle path
   
   OUTPUT : 
   - segment_length : length of muscle path """

   print("on fait l'algo avec celui ci = ", insertion_muscle)
   # First of all, create a rotation matrix (the model have y and not z for ax up) 
   # Single cylinder algo and double cylinders algo don't work without this change
   matrix_rot_zy = np.array([[0,0,1,0], [1,0,0,0], [0,1,0,0], [0,0,0,1]])

   # Rotation
   origin_muscle_rot = np.dot(matrix_rot_zy[0:3, 0:3], origin_muscle)
   insertion_muscle_rot = np.dot(matrix_rot_zy[0:3, 0:3], insertion_muscle)
   points = [origin_muscle_rot, insertion_muscle_rot]
   
   for cylinder in cylinders :
      cylinder.compute_new_matrix_segment(model, q) 
      cylinder.compute_matrix_rotation_zy(matrix_rot_zy) 
         
   for k in range(2) : 
      cylinders[k].compute_new_radius(points[k])
   
   if (len(cylinders) == 0) :
      # Muscle path is straight line from origin_point to final_point
      segment_length = norm(np.array(origin_muscle_rot) - np.array(insertion_muscle_rot))
      points_not_in_cylinder = []
      bool_inactive = []

   elif (len(cylinders) == 1) :

      Q_rot, T_rot, Q_T_inactive, segment_length = single_cylinder_obstacle_set_algorithm(origin_muscle_rot, insertion_muscle_rot, cylinders[0], plot_cadran)
      points_not_in_cylinder = [origin_muscle_rot, insertion_muscle_rot]
      bool_inactive = [Q_T_inactive]
      
      if plot == True : 
         plot_one_cylinder_obstacle(origin_muscle_rot, insertion_muscle_rot, cylinders[0], Q_rot, T_rot, Q_T_inactive)

   else : # (len(cylinders) == 2 ) 

      Q_rot, G_rot, H_rot, T_rot, Q_G_inactive, H_T_inactive , segment_length  = double_cylinder_obstacle_set_algorithm(origin_muscle_rot, insertion_muscle_rot, cylinders[0], cylinders[1], plot_cadran)
      points_not_in_cylinder = [origin_muscle_rot, H_rot, G_rot, insertion_muscle_rot]
      bool_inactive = [Q_G_inactive, H_T_inactive]
      
      if plot == True : 
         plot_double_cylinder_obstacle(origin_muscle_rot, insertion_muscle_rot, cylinders[0], cylinders[1], Q_rot, G_rot, H_rot, T_rot, Q_G_inactive, H_T_inactive)

   data_ignored = find_if_data_ignored(cylinders, points_not_in_cylinder, bool_inactive)
   
   print("data_ignored = ", data_ignored)
      
   return segment_length, data_ignored

