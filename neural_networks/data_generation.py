import numpy as np
from wrapping.algorithm import single_cylinder_obstacle_set_algorithm, double_cylinder_obstacle_set_algorithm
from wrapping.plot_cylinder import plot_one_cylinder_obstacle, plot_double_cylinder_obstacle
import pandas as pd
from scipy.linalg import norm
import matplotlib.pyplot as plt

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

def compute_segment_length(model, cylinders, q, origin_muscle, insertion_muscle, plot = False) :

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

   # First of all, create a rotation matrix (the model have y and not z for ax up) 
   # Single cylinder algo and double cylinders algo don't work without this change
   matrix_rot_zy = np.array([[0,0,1,0], [1,0,0,0], [0,1,0,0], [0,0,0,1]])

   # Rotation
   origin_muscle_rot = np.dot(matrix_rot_zy[0:3, 0:3], origin_muscle)
   insertion_muscle_rot = np.dot(matrix_rot_zy[0:3, 0:3], insertion_muscle)
   
   # Compute transformation of cylider matrix
   for cylinder in cylinders :
      cylinder.compute_new_matrix_segment(model, q)
      cylinder.compute_matrix_rotation_zy(matrix_rot_zy) 

   if (len(cylinders) == 0) :
      # Muscle path is straight line from origin_point to final_point
      segment_length = norm(np.array(origin_muscle_rot) - np.array(insertion_muscle_rot))

   elif (len(cylinders) == 1) :

      Q_rot, T_rot, Q_T_inactive, segment_length = single_cylinder_obstacle_set_algorithm(origin_muscle_rot, insertion_muscle_rot, cylinders[0])

      if plot == True : 
         plot_one_cylinder_obstacle(origin_muscle_rot, insertion_muscle_rot, cylinders[0], Q_rot, T_rot, Q_T_inactive)

   else : # (len(cylinders) == 2 ) 

      Q_rot, G_rot, H_rot, T_rot, Q_G_inactive, H_T_inactive , segment_length  = double_cylinder_obstacle_set_algorithm(origin_muscle_rot, insertion_muscle_rot, cylinders[0], cylinders[1], np.dot(np.linalg.inv(cylinders[1].matrix), cylinders[0].matrix) )

      if plot == True : 
         plot_double_cylinder_obstacle(origin_muscle_rot, insertion_muscle_rot, cylinders[0], cylinders[1], Q_rot, G_rot, H_rot, T_rot, Q_G_inactive, H_T_inactive)

   return segment_length  


def add_line_df(filename, muscle_selected_index, q, origin_muscle, insertion_muscle, segment_length) :
   """Add one line to the file
   
   INPUT : 
   - filename : string, name of the file to edit
   - muscle_selected_index : int, index of the muscle selected
   - q : array 4*2, q randomly generated
   - segment_length : length of muscle path (for this generated q)"""
   
   import os
   if not os.path.exists(filename):
      # Create excel file with initial structure
      data = {
      "muscle_selected": [],
      "humerus_right_RotY": [],
      "humerus_right_RotX": [],
      "humerus_right_RotY2": [],
      "ulna_effector_right_RotZ": [],
      "origin_muscle_x": [],
      "origin_muscle_y": [],
      "origin_muscle_z": [],
      "insertion_muscle_x": [],
      "insertion_muscle_y": [],
      "insertion_muscle_z": [],
      "segment_length": []
      }
      
      pd.DataFrame(data).to_excel(filename, index = False)

   # Read the existing data from the Excel file
   df = pd.read_excel(filename)
    
   # Create a new line with the provided data
   new_line = {
      "muscle_selected": muscle_selected_index,
      "humerus_right_RotY": q[0],
      "humerus_right_RotX": q[1],
      "humerus_right_RotY2": q[2],
      "ulna_effector_right_RotZ": q[3],
      "origin_muscle_x": origin_muscle[0],
      "origin_muscle_y": origin_muscle[1],
      "origin_muscle_z": origin_muscle[2],
      "insertion_muscle_x": insertion_muscle[0],
      "insertion_muscle_y": insertion_muscle[1],
      "insertion_muscle_z": insertion_muscle[2],
      "segment_length": segment_length
   }
   
   # Append the new line to the DataFrame using pd.concat
   df = pd.concat([df, pd.DataFrame([new_line])], ignore_index=True)
   
   with pd.ExcelWriter(filename, engine='openpyxl', mode='w') as writer:
      df.to_excel(writer, index=False)
         

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
   
   muscle_index = initialisation_generation(model, muscle_selected, cylinders)
 
   # Limits of q
   min_vals = [row[0] for row in q_ranges_muscle]
   max_vals = [row[1] for row in q_ranges_muscle] 

   for k in range (dataset_size) : 
      print("k = ", k)

      # Generate a random q 
      q = np.random.uniform(low=min_vals, high=max_vals)
      
      origin_muscle, insertion_muscle = update_points_position(model, muscle_index, q)
      
      # ------------------------------------------------

      segment_length = compute_segment_length(model, cylinders, q, origin_muscle, insertion_muscle, plot)  
   
      # Add line to data frame
      add_line_df(filename, muscle_index, q, origin_muscle, insertion_muscle, segment_length)

   return None


def test_limit_data_for_learning (muscle_selected, cylinders, model, q_ranges, filename, plot=False) :
   
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
            print("q = ", q)
            
            # Updates
            model.updateMuscles(q) 
            mus = model.muscle(muscle_index) 
            
            model.UpdateKinematicsCustom(q)

            # Find coordinates of origin point (P) and insertion point (S)
            origin_muscle = mus.musclesPointsInGlobal(model, q)[0].to_array() 
            insertion_muscle = mus.musclesPointsInGlobal(model, q)[-1].to_array() 
            
            print("origin = ", origin_muscle)
            print("insertion  = ", insertion_muscle)
            
            # ------------------------------------------------

            segment_length = compute_segment_length(model, cylinders, q, origin_muscle, insertion_muscle, plot)  
         
            # Add line to data frame
            add_line_df(f"test_limit_{filename}", muscle_index, q, origin_muscle, insertion_muscle, segment_length)

   return None

def data_for_learning_plot (muscle_selected, cylinders, model, q_ranges_muscle, i, num_points = 100, plot_all = False, plot_limit = False) :
   
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
   
   muscle_index = initialisation_generation(model, muscle_selected, cylinders)

   segment_lengths = []
   qs = []
   
   print("q range = ", q_ranges_muscle)

   for k in range (num_points+1) : 
      print("k = ", k)
      
      # Incrémenter qi
      qi = k * ((q_ranges_muscle[i][1] - q_ranges_muscle[i][0]) / num_points) + q_ranges_muscle[i][0]
      print("qi = ", qi)
      
      # Generate a random q 
      q = np.array([0.,0.,0.,0.])
      q[i] = qi
      
      origin_muscle, insertion_muscle = update_points_position(model, muscle_index, q)
      
      # ------------------------------------------------

      if k in [0,num_points/2, num_points] and plot_limit :
         segment_length = compute_segment_length(model, cylinders, q, origin_muscle, insertion_muscle, plot_limit)  
      else :  
         segment_length = compute_segment_length(model, cylinders, q, origin_muscle, insertion_muscle, plot_all)  
      
      qs.append(qi)
      segment_lengths.append(segment_length)
      

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

   return None