import numpy as np
from wrapping.algorithm import single_cylinder_obstacle_set_algorithm, double_cylinder_obstacle_set_algorithm
from wrapping.plot_cylinder import plot_one_cylinder_obstacle, plot_double_cylinder_obstacle
from scipy.linalg import norm
from openpyxl import load_workbook
from neural_networks.discontinuities import *
import copy

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

def ensure_unique_column_names(dofs_list) : 

   # Utilisation d'un dictionnaire pour compter les occurrences et générer les noms uniques
   name_counts = {}
   unique_names = []

   for name in dofs_list:
      if name in name_counts:
         name_counts[name] += 1
         unique_name = f"{name}_{name_counts[name]}"
      else:
         name_counts[name] = 0
         unique_name = name
      
      unique_names.append(unique_name)

   return unique_names


def compute_q_ranges(model):
   segment_names = [model.segment(i).name().to_string() for i in range(model.nbSegment())]
   # on calcule tous les q ranges de chaque articulation (2, 4, 6, 7)
   q_ranges = []
   q_ranges_names_with_dofs = []
   # Find name of segments and matrix of cylinders  
   
   for segment_index in range(len(segment_names)) :
      q_ranges_segment = [[ranges.min(), ranges.max()] for ranges in model.segment(segment_index).QRanges()]
      q_ranges+= q_ranges_segment
      humerus_dof_names = [model.segment(segment_index).nameDof(i).to_string() for i in 
                     range(model.segment(segment_index).nbQ())]
      q_ranges_names_with_dofs += [f"{segment_names[segment_index]}_{dof}" for dof in humerus_dof_names]
   q_ranges_names_with_dofs = ensure_unique_column_names(q_ranges_names_with_dofs)
   
   return q_ranges, q_ranges_names_with_dofs

def initialisation_generation(model, q_ranges, muscle_index, cylinders) :
   
   segment_names = [model.segment(i).name().to_string() for i in range(model.nbSegment())]

   # q_initial = np.array([0.,-0.01,0.,0.05])
   # q_initial = np.array([0.0, 0.0, 0.0, 0.0]) 
   q_initial = np.array([0.0 for i in range(len(q_ranges))])
   
   for cylinder in cylinders : 
      cylinder.compute_seg_index_and_gcs_seg_0(q_initial, model, segment_names)
   
   origin_muscle, insertion_muscle = update_points_position(model, [0, -1], muscle_index, q_initial) # global initial
   points = [origin_muscle, insertion_muscle]
   for k in range(len(cylinders)) : 
      cylinders[k].compute_new_radius(points[k])


def update_points_position(model, point_index, muscle_index, q) : 
   # Updates
   # en gros, on fait un update par rapport à un muscle avec muscle index et le q doit correspondre
   model.updateMuscles(q) 
   mus = model.muscle(muscle_index) 
   
   model.UpdateKinematicsCustom(q)

   # Find coordinates of origin point (P) and insertion point (S)
   points = []
   for k in (point_index) : 
      points.append(mus.musclesPointsInGlobal(model, q)[k].to_array()) 
   
   # ces points sont dans le repere global (verifiee)
   # cela revient à faire :
   # switch_frame([0.016, -0.0354957, 0.005], gcs_seg))
   
   return (points[k] for k in range (len(points)))
       

def find_index_muscle(model, muscle):
   
   """ Find index of the muscle selected in the name list
   
   INPUT : 
   - muscle : string, name of the muscle selected
   - muscle_names : string, list of muscle names
   
   OUTPUT : 
   - position : int, position of the muscle in the list"""

   # Find index of the muscle selected
   muscle_names = [model.muscle(i).name().to_string() for i in range(model.nbMuscles())]

   try:
      position = muscle_names.index(muscle) 
      return position
   except ValueError:
      valid_muscles = ", ".join(muscle_names)
      raise ValueError(f"Invalid muscle name '{muscle}'. You must choose a valid muscle from this list: [{valid_muscles}]")

def compute_segment_length(model, cylinders, muscle_index, q_ranges, q, origin_muscle, insertion_muscle, plot = False, plot_cadran = False) :

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

   initialisation_generation(model, q_ranges, muscle_index, cylinders)
   
   print("on fait l'algo avec celui ci insertion_muscle = ", insertion_muscle)
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
         
   for k in range(len(cylinders)) : 
      cylinders[k].compute_new_radius(points[k])
   
   if (len(cylinders) == 0) :
      # Muscle path is straight line from origin_point to final_point
      # Warning : is just a straight line, don't pass by via point !
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

