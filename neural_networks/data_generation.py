import numpy as np
from wrapping.algorithm import single_cylinder_obstacle_set_algorithm, double_cylinder_obstacle_set_algorithm
from wrapping.plot_cylinder import plot_one_cylinder_obstacle, plot_double_cylinder_obstacle
import pandas as pd
from scipy.linalg import norm
import matplotlib.pyplot as plt
import os
from openpyxl import load_workbook
from neural_networks.ExcelBatchWriter import ExcelBatchWriter

from wrapping.step_1 import *

def initialisation_generation(model, muscle_selected, cylinders) :
   # Find index of the muscle selected
   muscle_names = [model.muscle(i).name().to_string() for i in range(model.nbMuscles())]
   muscle_index = find_index_muscle(muscle_selected, muscle_names)

   # Find name of segments and matrix of cylinders  
   segment_names = [model.segment(i).name().to_string() for i in range(model.nbSegment())]
   q_initial = np.array([0.,-0.01,0.,0.05])
   for cylinder in cylinders : 
      cylinder.compute_seg_index_and_gcs_seg_0(q_initial, model, segment_names)
   
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
   
   # correction side humerus
   
   
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

def compute_segment_length(model, cylinders, q, origin_muscle, insertion_muscle, list_ref = [], plot = False) :

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
   
   # Compute transformation of cylider matrix
   # cylinders[0].compute_new_matrix_segment(model, q)
   # cylinders[0].compute_matrix_rotation_zy(matrix_rot_zy) 
   # cylinders[1].compute_new_matrix_segment2(model, q)
   # cylinders[1].compute_matrix_rotation_zy(matrix_rot_zy) 
   
   # vrai truc, en haut pas propre
   for cylinder in cylinders :
      cylinder.compute_new_matrix_segment(model, q)
      cylinder.compute_matrix_rotation_zy(matrix_rot_zy) 
   # cylinders[1].compute_new_matrix_segment(model, q)
   

   if (len(cylinders) == 0) :
      # Muscle path is straight line from origin_point to final_point
      segment_length = norm(np.array(origin_muscle_rot) - np.array(insertion_muscle_rot))

   elif (len(cylinders) == 1) :

      Q_rot, T_rot, Q_T_inactive, segment_length = single_cylinder_obstacle_set_algorithm(origin_muscle_rot, insertion_muscle_rot, cylinders[0])

      if plot == True : 
         plot_one_cylinder_obstacle(origin_muscle_rot, insertion_muscle_rot, cylinders[0], Q_rot, T_rot, Q_T_inactive)

   else : # (len(cylinders) == 2 ) 

      Q_rot, G_rot, H_rot, T_rot, Q_G_inactive, H_T_inactive , segment_length  = double_cylinder_obstacle_set_algorithm(origin_muscle_rot, insertion_muscle_rot, cylinders[0], cylinders[1], list_ref)

      if plot == True : 
         plot_double_cylinder_obstacle(origin_muscle_rot, insertion_muscle_rot, cylinders[0], cylinders[1], Q_rot, G_rot, H_rot, T_rot, Q_G_inactive, H_T_inactive)

   return Q_rot, G_rot, H_rot, T_rot, Q_G_inactive, H_T_inactive , segment_length  

def data_for_learning (muscle_selected, cylinders, model, q_ranges_muscle, dataset_size, filename, plot=False) :
   
   """Create a data frame for prepare datas
   
   INPUT
   - muscle_selected : string, name of the muscle selected. 
                        Please chose an autorized name in this list : 
                        ['PECM2', 'PECM3', 'LAT', 'DELT2', 'DELT3', 'INFSP', 'SUPSP', 'SUBSC', 'TMIN', 'TMAJ',
                        'CORB', 'TRIlong', 'PECM1', 'DELT1', 'BIClong', 'BICshort']
   - cylinders : List of muscle's cylinder (0, 1 or 2 cylinders)
   - model : model 
   - q_ranges_muscle : array 4*2, q ranges limited for the muscle selected 
   - dataset_size : int, number of data we would like
   - plot : bool (default false), True if we want a plot of point P, S (and Q, G, H and T) with cylinder(s)"""
   
   writer = ExcelBatchWriter(filename, batch_size=100)
   muscle_index = initialisation_generation(model, muscle_selected, cylinders)
 
   # Limits of q
   min_vals = [row[0] for row in q_ranges_muscle]
   max_vals = [row[1] for row in q_ranges_muscle] 

   for k in range (dataset_size) : 
      print("k = ", k)

      # Generate a random q 
      q = np.random.uniform(low=min_vals, high=max_vals)
      
      origin_muscle, insertion_muscle = update_points_position(model, muscle_index, q)
      
      # # essai 
      # for cylinder in cylinders : 
      #    if cylinder.segment == "humerus_right" : 
      #       cylinder.correcte_side(q)
      
      # ------------------------------------------------

      segment_length = compute_segment_length(model, cylinders, q, origin_muscle, insertion_muscle, plot)  
   
      # Add line to data frame
      # add_line_df(filename, muscle_index, q, origin_muscle, insertion_muscle, segment_length)
      writer.add_line(muscle_index, q, origin_muscle, insertion_muscle, segment_length)
   
   # Ensure remaining lines are written to file
   writer.close()
   
   return None

# def data_for_learning (muscle_selected, cylinders, model, q_ranges_muscle, dataset_size, filename, plot=False) :
   
#    """Create a data frame for prepare datas
   
#    INPUT
#    - muscle_selected : string, name of the muscle selected. 
#                         Please chose an autorized name in this list : 
#                         ['PECM2', 'PECM3', 'LAT', 'DELT2', 'DELT3', 'INFSP', 'SUPSP', 'SUBSC', 'TMIN', 'TMAJ',
#                         'CORB', 'TRIlong', 'PECM1', 'DELT1', 'BIClong', 'BICshort']
#    - cylinders : List of muscle's cylinder (0, 1 or 2 cylinders)
#    - model : model 
#    - q_ranges_muscle : array 4*2, q ranges limited for the muscle selected 
#    - dataset_size : int, number of data we would like
#    - plot : bool (default false), True if we want a plot of point P, S (and Q, G, H and T) with cylinder(s)"""
   
#    writer = ExcelBatchWriter(filename, batch_size=100)
#    muscle_index = initialisation_generation(model, muscle_selected, cylinders)
 
#    # Limits of q
#    min_vals = [row[0] for row in q_ranges_muscle]
#    max_vals = [row[1] for row in q_ranges_muscle] 

#    k = 0
#    while k < dataset_size : 
#       print("k = ", k)

#       # Generate a random q 
#       q = np.random.uniform(low=min_vals, high=max_vals)
      
#       origin_muscle, insertion_muscle = update_points_position(model, muscle_index, q)
      
#       # # essai 
#       # for cylinder in cylinders : 
#       #    if cylinder.segment == "humerus_right" : 
#       #       cylinder.correcte_side(q)
      
#       # ------------------------------------------------

#       segment_length = compute_segment_length(model, cylinders, q, origin_muscle, insertion_muscle, plot)  
   
#       # Add line to data frame
#       # add_line_df(filename, muscle_index, q, origin_muscle, insertion_muscle, segment_length)
#       writer.add_line(muscle_index, q, origin_muscle, insertion_muscle, segment_length)
      
#       # Ensure remaining lines are written to file
#       writer.close()


   
#    return None


def test_limit_data_for_learning (muscle_selected, cylinders, model, q_ranges, plot=False) :
   
   """Test limits of q for the muscle selected
   
   INPUT
   - muscle_selected : string, name of the muscle selected. 
                        Please chose an autorized name in this list : 
                        ['PECM2', 'PECM3', 'LAT', 'DELT2', 'DELT3', 'INFSP', 'SUPSP', 'SUBSC', 'TMIN', 'TMAJ',
                        'CORB', 'TRIlong', 'PECM1', 'DELT1', 'BIClong', 'BICshort']
   - cylinders : List of muscle's cylinder (0, 1 or 2 cylinders)
   - model : model 
   - q_ranges_muscle : array 4*2, q ranges limited for the muscle selected 
   - dataset_size : int, number of data we would like
   - plot : bool (default false), True if we want a plot of point P, S (and Q, G, H and T) with cylinder(s)
   - test : bool (default false), True if we just testing (for example, to choose q_ranges_muscle)"""
   
   muscle_index = initialisation_generation(model, muscle_selected, cylinders)

   q_test_limite = [[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]]
   for k in range (3) :
      q_test_limite[k][0]  = q_ranges [k][0] 
      q_test_limite[k][1]  = (q_ranges [k][0]  + q_ranges[k][1] ) /2
      q_test_limite[k][2]  = q_ranges[k][1] 

   for i in range (3) :
      for j in range (3) :
         for k in range (3) :
            
            print("i = ", i, " j = ", j, " k = ", k)
            
            q = np.array([q_test_limite[0][i],q_test_limite[1][j], q_test_limite[2][k], 0])
            # q = np.array([-0.736696402602619,-2.81278701317883, 1.56632461528205, 0])
            print("q = ", q)
            
            
            origin_muscle, insertion_muscle = update_points_position(model, muscle_index, q)
            
            # for cylinder in cylinders : 
            #    if cylinder.segment == "humerus_right" : 
            #       cylinder.correcte_side(q)
            
            print("origin = ", origin_muscle)
            print("insertion  = ", insertion_muscle)
            
            # ------------------------------------------------

            segment_length = compute_segment_length(model, cylinders, q, origin_muscle, insertion_muscle, [], plot)  
            print("segment_length = ", segment_length)

   return None

def data_for_learning_plot (filename, muscle_selected, cylinders, model, q_ranges_muscle, q_fixed, i, num_points = 100, plot_all = False, plot_limit = False) :
   
   """Create a data frame for prepare datas
   
   INPUT
   - muscle_selected : string, name of the muscle selected. 
                        Please chose an autorized name in this list : 
                        ['PECM2', 'PECM3', 'LAT', 'DELT2', 'DELT3', 'INFSP', 'SUPSP', 'SUBSC', 'TMIN', 'TMAJ',
                        'CORB', 'TRIlong', 'PECM1', 'DELT1', 'BIClong', 'BICshort']
   - cylinders : List of muscle's cylinder (0, 1 or 2 cylinders)
   - model : model 
   - q_ranges_muscle : array 4*2, q ranges limited for the muscle selected 
   - dataset_size : int, number of data we would like
   - plot : bool (default false), True if we want a plot of point P, S (and Q, G, H and T) with cylinder(s)"""
   
   writer = ExcelBatchWriter(filename, batch_size=100)
   muscle_index = initialisation_generation(model, muscle_selected, cylinders)

   q_ref = np.array([q_ranges_muscle[0][1], q_ranges_muscle[1][1], q_ranges_muscle[2][1], 0.0]) #
   # q_ref = np.array([0,0,0,0])
   # # q ref pour le moment, tout au max de range 
   # matrix_U = np.array([[ 0.96734723,  0.24692246, -0.05693821, -0.02614252],
   #                      [-0.24253723,  0.96726508,  0.07273017, -0.00298349],
   #                      [ 0.07339071, -0.05744761,  0.99580371,  0.17122542],
   #                      [ 0.        ,  0.        ,  0.        ,  1.        ]])
   
   # matrix_V = np.array([[ 0.96903113, -0.24650421,  0.01464025, -0.03536606],
   #                      [ 0.    	, -0.05928701, -0.99824098, -0.03672373],
   #                      [ 0.24693858,  0.96732659, -0.05745096,  0.1813522 ],
   #                      [ 0.    	,  0.    	,  0.    	,  1.    	]])

   # point_insertion = switch_frame_UV([0.016, -0.0354957, 0.005], matrix_U, matrix_V)

   origin_muscle, insertion_muscle = update_points_position(model, muscle_index, q_ref)
   Q_ref, G_ref, H_ref, T_ref, _, _ , _ = compute_segment_length(model, cylinders, q_ref, origin_muscle, insertion_muscle, [], False)
   # Q, G, H, T ref sont dans le repere global (si j'ai pas fait de betises)
   list_ref = [Q_ref, G_ref, H_ref, T_ref]
   
   matrix_rot_yzc = np.array([[0,1,0,0], [0,0,1,0], [1,0,0,0], [0,0,0,1]])
   # # Homogeneisation
   for n in range (len(list_ref)) : 
   #    list_ref[i] = np.append(list_ref[i],1)
      list_ref[n] = np.dot(matrix_rot_yzc[0:3, 0:3], list_ref[n])
   
   # list_ref = []
   
   q = q_fixed
   
   segment_lengths = []
   qs = []
   
   print("q range = ", q_ranges_muscle)

   for k in range (num_points+1) : 
      print("k = ", k)
      
      # Incrémenter qi
      qi = k * ((q_ranges_muscle[i][1] - q_ranges_muscle[i][0]) / num_points) + q_ranges_muscle[i][0]
      print("qi = ", qi)
      
      q[i] = qi
      
      print("q = ", q)
      
      origin_muscle, insertion_muscle = update_points_position(model, muscle_index, q)
      
      # for cylinder in cylinders : 
      #    if cylinder.segment == "humerus_right" : 
      #       cylinder.correcte_side(q)
      
      # ------------------------------------------------

      if k in [0,num_points/2, num_points] and plot_limit :
         _, _, _, _, _, _ , segment_length   = compute_segment_length(model, cylinders, q, origin_muscle, insertion_muscle, list_ref, plot_limit)  
         
      else :  
         _, _, _, _, _, _ , segment_length   = compute_segment_length(model, cylinders, q, origin_muscle, insertion_muscle, list_ref, plot_all)  
      
      print("segment_length = ", segment_length)
      qs.append(qi)
      segment_lengths.append(segment_length)
      

      
      writer.add_line(muscle_index, q, origin_muscle, insertion_muscle, segment_length)

   # Création du graphique
   plt.plot(qs, segment_lengths, marker='o', linestyle='-', color='b')

   # Ajout des étiquettes et du titre
   plt.xlabel(f'q{i}')
   plt.ylabel('Muscle_length')
   plt.title(f'Muscle Length as a Function of q{i} Values')

   # Définir les marques sur l'axe x et y de manière espacée
   plt.xticks(qs[::5])  # Afficher les ticks tous les 5 sur l'axe x
   plt.yticks(segment_lengths[::5])  # Afficher les ticks tous les 10 sur l'axe y
   
   # Affichage de la grille
   plt.grid(True)
   plt.show()
   
   writer.close()
   
   return None