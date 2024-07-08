import numpy as np
import matplotlib.pyplot as plt
from neural_networks.ExcelBatchWriter import ExcelBatchWriter
from neural_networks.discontinuities import *
from neural_networks.functions_data_generation import *
from neural_networks.plot_visualisation import plot_mvt_discontinuities_in_red
from neural_networks.file_directory_operations import create_directory
import copy
import os

def data_for_learning_ddl (muscle_selected, cylinders, model, dataset_size, filename, data_without_error = False, plot=False, plot_cadran = False) :
   
   """Create a data frame for prepare datas
   Datas are generated ponctually, independantly and uniformly
   
   INPUT
   - muscle_selected : string, name of the muscle selected. 
                        Please chose an autorized name in this list : 
                        ['PECM2', 'PECM3', 'LAT', 'DELT2', 'DELT3', 'INFSP', 'SUPSP', 'SUBSC', 'TMIN', 'TMAJ',
                        'CORB', 'TRIlong', 'PECM1', 'DELT1', 'BIClong', 'BICshort']
   - cylinders : List of muscle's cylinder (0, 1 or 2 cylinders)
   - model : model 
   - q_ranges_muscle : array 4*2, q ranges limited for the muscle selected 
   - dataset_size : int, number of data we would like
   - filename : string, name of the file to create
   - num_points : int (default = 50) number of point to generate per mvt
   - data_without_error : bool (default = False), True to ignore data with error wrapping 
   - plot : bool (default false), True if we want a plot of point P, S (and Q, G, H and T) with cylinder(s)
   - plot_cradran : bool (default = False), True to show cadran, pov of each cylinder and wrapping"""
   
   q_ranges, q_ranges_names_with_dofs = compute_q_ranges(model)
   muscle_index = initialisation_generation(model, q_ranges, muscle_selected, cylinders)
       
   # Limits of q
   min_vals = [row[0] for row in q_ranges]
   max_vals = [row[1] for row in q_ranges] 
   
   writer = ExcelBatchWriter(filename+"xlsx", q_ranges_names_with_dofs, batch_size=100)

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

def plot_one_q_variation(muscle_selected, cylinders, model, q_fixed, i, filename, num_points = 100, plot_all = False, plot_limit = False, plot_cadran=False) :
   
   """Create a directory with an excel file for one q and png of mvt
   
   INPUT
   - muscle_selected : string, name of the muscle selected. 
                        Please chose an autorized name in this list : 
                        ['PECM2', 'PECM3', 'LAT', 'DELT2', 'DELT3', 'INFSP', 'SUPSP', 'SUBSC', 'TMIN', 'TMAJ',
                        'CORB', 'TRIlong', 'PECM1', 'DELT1', 'BIClong', 'BICshort']
   - cylinders : List of muscle's cylinder (0, 1 or 2 cylinders)
   - model : model 
   - q_ranges_muscle : array 4*2, q ranges limited for the muscle selected 
   - q_fixed : array 4*1, q fixed, reference
   - i : int (0, 1, 2, 3), qi to do variate
   - filename : string, name of the file to create
   - num_points : int (default = 50) number of point to generate per mvt
   - plot_all : bool (default false), True if we want all plots of point P, S (and Q, G, H and T) with cylinder(s)
   - plot_limit : bool (default = False), True to plot points P, S (and Q, G, H and T) with cylinder(s) 
                                                                                          (first, middle and last one)
   - plot_cradran : bool (default = False), True to show cadran, pov of each cylinder and wrapping"""

   q_ranges, q_ranges_names_with_dofs = compute_q_ranges(model)
   muscle_index= initialisation_generation(model, q_ranges, muscle_selected, cylinders)
   
   directory = "plot_one_q_variation_" + filename
   create_directory(directory)
   writer = ExcelBatchWriter(f"{directory}/"+filename+".xlsx", q_ranges_names_with_dofs, batch_size=100)

   q = q_fixed
   
   segment_lengths = []
   qs = []
   
   print("q range = ", q_ranges)

   for k in range (num_points+1) : 
      print("k = ", k)
      
      qi = k * ((q_ranges[i][1] - q_ranges[i][0]) / num_points) + q_ranges[i][0]
      q[i] = qi
      
      print("q = ", q)
      
      origin_muscle, insertion_muscle = update_points_position(model, muscle_index, q)

      # ------------------------------------------------

      if k in [0,num_points/2, num_points] and plot_limit :
         segment_length, _ = compute_segment_length(model, cylinders, q, origin_muscle, insertion_muscle, plot_limit, plot_cadran)  
         
      else :  
         segment_length, _ = compute_segment_length(model, cylinders, q, origin_muscle, insertion_muscle, plot_all, plot_cadran)  
      
      print("segment_length = ", segment_length)
      qs.append(qi)
      segment_lengths.append(segment_length)
      
      writer.add_line(muscle_index, q, origin_muscle, insertion_muscle, segment_length)

   plt.plot(qs, segment_lengths, marker='o', linestyle='-', color='b')
   plt.xlabel(f'q{i}')
   plt.ylabel('Muscle_length')
   plt.title(f'Muscle Length as a Function of q{i} Values')
   plt.xticks(qs[::5])
   plt.yticks(segment_lengths[::5]) 
   plt.grid(True)
   plt.savefig(f"{directory}/one_q_variation.png")
   plt.show()
   
   find_discontinuty(qs, segment_lengths, plot_discontinuities=True)
   
   writer.close()
   return None

def plot_all_q_variation(muscle_selected, cylinders, model, q_fixed, filename, num_points = 100, plot_all = False, plot_limit = False, plot_cadran=False) :
   
   """Create a directory with all excel files and png of mvt for all q
   
   INPUT
   - muscle_selected : string, name of the muscle selected. 
                        Please chose an autorized name in this list : 
                        ['PECM2', 'PECM3', 'LAT', 'DELT2', 'DELT3', 'INFSP', 'SUPSP', 'SUBSC', 'TMIN', 'TMAJ',
                        'CORB', 'TRIlong', 'PECM1', 'DELT1', 'BIClong', 'BICshort']
   - cylinders : List of muscle's cylinder (0, 1 or 2 cylinders)
   - model : model 
   - q_fixed : array 4*1, q fixed, reference
   - filename : string, name of the file to create
   - num_points : int (default = 50) number of point to generate per mvt
   - plot_all : bool (default false), True if we want all plots of point P, S (and Q, G, H and T) with cylinder(s)
   - plot_limit : bool (default = False), True to plot points P, S (and Q, G, H and T) with cylinder(s) 
                                                                                          (first, middle and last one)
   - plot_cradran : bool (default = False), True to show cadran, pov of each cylinder and wrapping"""
   
   # Create a folder for save excel files and plots
   directory = "plot_all_q_variation_" + filename
   create_directory(directory)
   
   q_ranges, q_ranges_names_with_dofs = compute_q_ranges(model)
   muscle_index= initialisation_generation(model, q_ranges, muscle_selected, cylinders)
   q = q_fixed
   
   print("q range = ", q_ranges)

   fig, axs = plt.subplots(2, (len(q_ranges)+1)//2, figsize=(15, 10))

   for q_index in range (len(q_ranges)) : 
      segment_lengths = []
      qs = []
      writer = ExcelBatchWriter(f"{directory}/{q_index}_" + filename+".xlsx", q_ranges_names_with_dofs, batch_size=100)
      
      for k in range (num_points+1) : 
         print("k = ", k)
         
         qi = k * ((q_ranges[q_index][1] - q_ranges[q_index][0]) / num_points) + q_ranges[q_index][0]
         q[q_index] = qi
         
         print("q = ", q)
         
         origin_muscle, insertion_muscle = update_points_position(model, muscle_index, q)

         # ------------------------------------------------

         if k in [0,num_points/2, num_points] and plot_limit :
            segment_length, _ = compute_segment_length(model, cylinders, q, origin_muscle, insertion_muscle, plot_limit, plot_cadran)  
            
         else :  
            segment_length, _ = compute_segment_length(model, cylinders, q, origin_muscle, insertion_muscle, plot_all, plot_cadran)  
         
         print("segment_length = ", segment_length)
         qs.append(qi)
         segment_lengths.append(segment_length)
         

         writer.add_line(muscle_index, q, origin_muscle, insertion_muscle, segment_length)
      
      discontinuities = find_discontinuty(qs, segment_lengths, plot_discontinuities=False)
      
      row = q_index // ((len(q_ranges) + 1) // 2)
      col = q_index % ((len(q_ranges) + 1) // 2)

      axs[row, col].plot(qs, segment_lengths, marker='o', linestyle='-', color='b', markersize=3)
      for idx in discontinuities:
         axs[row, col].plot(qs[idx:idx+2], segment_lengths[idx:idx+2], 'r', linewidth=2)  # Discontinuities are in red
      axs[row, col].set_xlabel(f'q{q_index} Variation',fontsize='smaller')
      axs[row, col].set_ylabel('Muscle_length (m)',fontsize='smaller')
      axs[row, col].set_title(f'{q_ranges_names_with_dofs[q_index]}',fontsize='smaller')
      axs[row, col].set_xticks(qs[::5])
      axs[row, col].set_xticklabels([f'{x:.4f}' for x in qs[::5]],fontsize='smaller')

      axs[row, col].set_yticks(segment_lengths[::5])
      axs[row, col].set_yticklabels([f'{y:.4f}' for y in segment_lengths[::5]],fontsize='smaller')
   
   writer.close()
   
   fig.suptitle(f'Muscle Length as a Function of q Values\nq_fixed = {q_fixed}', fontweight='bold')
   plt.tight_layout()  
   plt.savefig(f"{directory}/all_q_variation.png")
   plt.show()
   
   return None

def data_for_learning_without_discontinuites_ddl(muscles_selected, cylinders, model, dataset_size, filename, num_points = 50, plot_cylinder_3D=False, plot_discontinuities = False, plot_cadran = False, plot_graph = False) :
   
   """
   NON FONCTIONNELLE
   Create a data frame for prepare datas without any discontinuities or error wrapping 
   Generate a mvt and then, remove problematic datas
   
   INPUT
   - muscle_selected : string, name of the muscle selected. 
                        Please chose an autorized name in this list : 
                        ['PECM2', 'PECM3', 'LAT', 'DELT2', 'DELT3', 'INFSP', 'SUPSP', 'SUBSC', 'TMIN', 'TMAJ',
                        'CORB', 'TRIlong', 'PECM1', 'DELT1', 'BIClong', 'BICshort']
   - cylinders : List of muscle's cylinder (0, 1 or 2 cylinders)
   - model : model 
   - q_ranges_muscle : array 4*2, q ranges limited for the muscle selected 
   - dataset_size : int, number of data we would like
   - filename : string, name of the file to create
   - num_points : int (default = 50) number of point to generate per mvt
   - plot : bool (default false), True if we want a plot of point P, S (and Q, G, H and T) with cylinder(s)
   - plot_discontinuities : bool (default = False), true to show mvt with discontinuity
   - plot_cradran : bool (default = False), True to show cadran, pov of each cylinder and wrapping"""
   
   q_ranges, q_ranges_names_with_dofs = compute_q_ranges(model)
   muscles_index = []
   for idx in range (len(muscles_selected)) : 
      muscles_index.append(initialisation_generation(model, q_ranges, muscles_selected[idx], cylinders[idx]))
   writer = ExcelBatchWriter(filename+".xlsx", q_ranges_names_with_dofs, batch_size=100)
 
   # Limits of q
   min_vals = [row[0] for row in q_ranges]
   max_vals = [row[1] for row in q_ranges] 

   num_line = 0
   while num_line < dataset_size : 
      print("num line = ", num_line)

      # Generate a random q 
      q_ref = np.random.uniform(low=min_vals, high=max_vals)
      
      for i in range (len(q_ranges)) :
         print("i = ", i)
         
         qs = []
         segment_lengths = []
         origin_muscles = [0 for n in range(len(muscles_index))]
         insertion_muscles = [0 for n in range(len(muscles_index))]
         datas_ignored = []
         lines = []
         to_remove = []
         
         q = copy.deepcopy(q_ref)
         
         # Generate points (num_points) of a mvt relatively to qi
         for j in range (num_points) : 
            print("j = ", j)
            
            # Incrémenter qi
            qi = j * ((q_ranges[i][1] - q_ranges[i][0]) / num_points) + q_ranges[i][0]
            q[i] = qi
            print("q = ", q)
            
            for k in range(len(muscles_index)) : 
         
               origin_muscles[k], insertion_muscles[k] = update_points_position(model, muscles_index[k], q)
               
               segment_length, data_ignored = compute_segment_length(model, cylinders, q, origin_muscles[k], insertion_muscles[k], plot_cylinder_3D, plot_cadran)  
               
               qs.append(qi)
               segment_lengths.append(segment_length)
               datas_ignored.append(data_ignored)
               
               lines.append([muscle_index, q, origin_muscle, insertion_muscle, segment_length])
         
         # Find indexes with discontinuties
         discontinuities = find_discontinuty(qs, segment_lengths, plot_discontinuities = plot_discontinuities)
         for discontinuity in discontinuities : 
            # min, max = data_to_remove_range(discontinuity, num_points, 5)
            min, max = data_to_remove_part(discontinuity, qs, num_points, 5)
            to_remove.extend(range(min, max + 1))
         positions = [n for n, ignored in enumerate(datas_ignored) if ignored]
         if len(positions) != 0 : 
            min, max = find_discontinuities_from_error_wrapping_range(positions, num_points, range = 5)
            to_remove.extend(range(min, max + 1))
         
         # Sort to keep only one occurancy of each indexes
         to_remove = sorted(set(to_remove), reverse=True)
         
         for index in to_remove :
            del lines[index]
         
         if len(to_remove) != 0 and plot_graph : 
            # To check points removed (in red)
            plot_mvt_discontinuities_in_red(i, qs, segment_lengths, to_remove)
         
         # Add lines
         for line in lines:
            writer.add_line(*line)
            num_line+=1
            if num_line > dataset_size : 
               break

   writer.close()
   return None