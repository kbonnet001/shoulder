import numpy as np
import matplotlib.pyplot as plt
from neural_networks.ExcelBatchWriter import ExcelBatchWriter
from neural_networks.discontinute import *
from neural_networks.functions_data_generation import *
import copy
import pandas as pd

def data_for_learning (muscle_selected, cylinders, model, q_ranges_muscle, dataset_size, filename, data_without_error = False, plot=False, plot_cadran = False) :
   
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

   k = 0
   while k < dataset_size : 
      print("k = ", k)

      # Generate a random q 
      q = np.random.uniform(low=min_vals, high=max_vals)
      
      origin_muscle, insertion_muscle = update_points_position(model, muscle_index, q)
      
      # ------------------------------------------------

      segment_length, data_ignored = compute_segment_length(model, cylinders, q, origin_muscle, insertion_muscle, plot, plot_cadran)  
   
      if (data_ignored == False and data_without_error == True) or (data_without_error == False) : 
         # Add line to data frame
         # add_line_df(filename, muscle_index, q, origin_muscle, insertion_muscle, segment_length)
         writer.add_line(muscle_index, q, origin_muscle, insertion_muscle, segment_length)
         print("hop dans le sheet")
         
         k+=1

   # Ensure remaining lines are written to file
   writer.close()
   
   return None

def test_limit_data_for_learning (muscle_selected, cylinders, model, q_ranges, plot=False, plot_cadran=False) :
   
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
            
            origin_muscle, insertion_muscle = update_points_position(model, muscle_index, q)
            
            # ------------------------------------------------

            segment_length, _ = compute_segment_length(model, cylinders, q, origin_muscle, insertion_muscle, plot, plot_cadran)  
            print("segment_length = ", segment_length)

   return None

def data_for_learning_plot (filename, muscle_selected, cylinders, model, q_ranges_muscle, q_fixed, i, num_points = 100, plot_all = False, plot_limit = False, plot_cadran=False) :
   
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

   origin_muscle, insertion_muscle = update_points_position(model, muscle_index, q_ref) # global
   
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
      
      # q = np.array([ 2.35619449, -1.49725651,  0.76039816 , 1.20305   ])
      print("q = ", q)
      
      origin_muscle, insertion_muscle = update_points_position(model, muscle_index, q)

      # ------------------------------------------------

      if k in [0,num_points/2, num_points] and plot_limit :
         segment_length, data_ignored = compute_segment_length(model, cylinders, q, origin_muscle, insertion_muscle, plot_limit, plot_cadran)  
         
      else :  
         segment_length, data_ignored = compute_segment_length(model, cylinders, q, origin_muscle, insertion_muscle, plot_all, plot_cadran)  
      
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
   
   find_discontinute(qs, segment_lengths)
   
   writer.close()
   
   return None

def data_for_learning_without_discontinuites(muscle_selected, cylinders, model, q_ranges_muscle, dataset_size, filename, num_points = 50, data_without_error = False, plot=False, plot_cadran = False) :
   
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

   num_line = 0
   while num_line < dataset_size : 
      print("num line = ", num_line)

      # Generate a random q 
      q_ref = np.random.uniform(low=min_vals, high=max_vals)
      
      for i in range (3) :
         print("i = ", i)
         
         qs = []
         segment_lengths = []
         datas_ignored = []
         lines = []
         to_remove = []
         
         q = copy.deepcopy(q_ref)
         
         for k in range (num_points) : 
            print("k = ", k)
            
            # Incrémenter qi
            qi = k * ((q_ranges_muscle[i][1] - q_ranges_muscle[i][0]) / num_points) + q_ranges_muscle[i][0]
            print("qi = ", qi)
            
            q[i] = qi
            
            print("q = ", q)
         
            origin_muscle, insertion_muscle = update_points_position(model, muscle_index, q)
            
            segment_length, data_ignored = compute_segment_length(model, cylinders, q, origin_muscle, insertion_muscle, plot, plot_cadran)  
            
            qs.append(qi)
            segment_lengths.append(segment_length)
            datas_ignored.append(data_ignored)
            
            lines.append([muscle_index, q, origin_muscle, insertion_muscle, segment_length])
            
         # on a fini de faire la courbe
         
         # on cherche maintenant les emplacements de discontinuite
         discontinuites = find_discontinute(qs, segment_lengths)
         for discontinuite in discontinuites : 
            min, max = data_to_remove(discontinuite, num_points, 5)
            to_remove.extend(range(min, max + 1))
         position = [n for n, ignored in enumerate(datas_ignored) if ignored]
         if len(position) != 0 : 
            min, max = find_discontinuites_from_error_wrapping(position, num_points, range = 5)
            to_remove.extend(range(min, max + 1))
         
         # tri pour ne garder aue les ligne à supprimer et ne pas avoir de doublon
         to_remove = sorted(set(to_remove), reverse=True)
         
         for index in to_remove :
            del lines[index]
         
         # ajouter les lignes dans le sheet
         for line in lines:
            writer.add_line(*line)
            num_line+=1
            if num_line > dataset_size : 
               break

   # Ensure remaining lines are written to file
   writer.close()
   
   return None