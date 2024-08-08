import numpy as np
import matplotlib.pyplot as plt
from neural_networks.ExcelBatchWriter import ExcelBatchWriter
from neural_networks.discontinuities import *
from neural_networks.functions_data_generation import *
from neural_networks.file_directory_operations import create_directory, create_and_save_plot
import copy
import random
from wrapping.muscles_length_jacobian import compute_dlmt_dq, plot_length_jacobian
from wrapping.muscle_forces_and_torque import compute_fm, compute_torque, compute_fm_muscle_index
import os
from neural_networks.other import compute_row_col, plot_mvt_discontinuities_in_red
from neural_networks.ExcelBatchWriterWithNoise import ExcelBatchWriterWithNoise
import pandas as pd

def plot_one_q_variation(muscle_selected, cylinders, model, q_fixed, i, filename, num_points = 100, plot_all = False, plot_limit = False, plot_cadran=False) :
   
   """Create a directory with an excel file for one q and png of mvt
   
   INPUTS
   - muscle_selected : string, name of the muscle selected. 
                        Please chose an autorized name in this list : 
                        ['PECM2', 'PECM3', 'LAT', 'DELT2', 'DELT3', 'INFSP', 'SUPSP', 'SUBSC', 'TMIN', 'TMAJ',
                        'CORB', 'TRIlong', 'PECM1', 'DELT1', 'BIClong', 'BICshort']
   - cylinders : List of muscle's cylinder (0, 1 or 2 cylinders)
   - model : model 
   - q_fixed : array 4*1, q fixed, reference
   - i : int (0, 1, 2, 3), qi to do variate
   - filename : string, name of the file to create
   - num_points : int (default = 50) number of point to generate per mvt
   - plot_all : bool (default false), True if we want all plots of point P, S (and Q, G, H and T) with cylinder(s)
   - plot_limit : bool (default = False), True to plot points P, S (and Q, G, H and T) with cylinder(s) 
                                                                                          (first, middle and last one)
   - plot_cradran : bool (default = False), True to show cadran, pov of each cylinder and wrapping"""

   q_ranges, q_ranges_names_with_dofs = compute_q_ranges(model)
   muscle_index= find_index_muscle(model, muscle_selected)
   initialisation_generation(model, muscle_index, cylinders)
   
   directory = "plot_one_q_variation_" + filename
   create_directory(directory)
   writer = ExcelBatchWriter(f"{directory}/"+filename+".xlsx", q_ranges_names_with_dofs, batch_size=100)

   q = q_fixed
   segment_lengths = []
   qs = []

   for k in range (num_points+1) : 
      print("plot one q variation, k = ", k)
      
      # q, qdot, alpha
      qi = k * ((q_ranges[i][1] - q_ranges[i][0]) / num_points) + q_ranges[i][0]
      q[i] = qi
      qdot = np.array([0.0 for _ in range(model.nbQ())]) # default
      alpha = 1 # default
      
      origin_muscle, insertion_muscle = update_points_position(model, [0, -1], muscle_index, q)

      # compute lmt
      if k in [0,num_points/2, num_points] and plot_limit :
         segment_length, _ = compute_segment_length(model, cylinders, q, origin_muscle, insertion_muscle, plot_limit, plot_cadran)  
         
      else :  
         segment_length, _ = compute_segment_length(model, cylinders, q, origin_muscle, insertion_muscle, plot_all, plot_cadran)  
      
      # compute others y
      qs.append(qi)
      segment_lengths.append(segment_length)
      dlmt_dq = compute_dlmt_dq(model, q, cylinders, muscle_index, delta_qi = 1e-8)
      muscle_force = compute_fm_muscle_index(model, muscle_index, q, qdot, alpha)
      torque, _ = compute_torque(dlmt_dq, muscle_force)
      
      # add line
      writer.add_line(muscle_index, q, qdot, alpha, origin_muscle, insertion_muscle, segment_length, copy.deepcopy(dlmt_dq), muscle_force, torque)

   # plot
   plt.plot(qs, segment_lengths, marker='o', linestyle='-', color='b')
   plt.xlabel(f'q{i}')
   plt.ylabel('Muscle_length')
   plt.title(f'Muscle Length as a Function of q{i} Values')
   plt.xticks(qs[::5])
   plt.yticks(segment_lengths[::5]) 
   plt.grid(True)
   create_and_save_plot(f"{directory}", "one_q_variation.png")
   plt.show()
   
   find_discontinuty(qs, segment_lengths, plot_discontinuities=True)
   
   writer.close()
   return None

def create_all_q_variation_files(muscle_selected, cylinders, model, q_fixed, filename, num_points = 100, plot_all = False, plot_limit = False, plot_cadran=False, file_path="") :
   
   """Create a directory with all excel files for all q (with a q_fixed)
   
   INPUTS
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
   - plot_cradran : bool (default = False), True to show cadran, pov of each cylinder and wrapping
   - file_path : path for the directory
   """
   
   # Create a folder for save excel files and plots
   directory = file_path+"/plot_all_q_variation_" + filename
   create_directory(directory)
   
   q_ranges, q_ranges_names_with_dofs = compute_q_ranges(model)
   muscle_index= find_index_muscle(model, muscle_selected)
   initialisation_generation(model, muscle_index, cylinders)
   q = copy.deepcopy(q_fixed)

   for q_index in range (model.nbQ()) : 
      if os.path.exists(f"{directory}/{q_index}_{q_ranges_names_with_dofs[q_index]}_" + filename+".xlsx") == False :

         writer = ExcelBatchWriter(f"{directory}/{q_index}_{q_ranges_names_with_dofs[q_index]}_" + filename+".xlsx", 
                                 q_ranges_names_with_dofs, batch_size=100)
         segment_lengths = []
         qs = []
         q = copy.deepcopy(q_fixed)
         
         for k in range (num_points+1) : 
            print("plot all q variation, k = ", k)
            
            # q, qdot, alpha
            qi = k * ((q_ranges[q_index][1] - q_ranges[q_index][0]) / num_points) + q_ranges[q_index][0]
            q[q_index] = qi
            
            qdot = np.array([0.0 for _ in range(model.nbQ())])
            alpha = 1
            
            origin_muscle, insertion_muscle = update_points_position(model, [0, -1], muscle_index, q)

            # compute lmt
            if k in [0,num_points/2, num_points] and plot_limit :
               segment_length, _ = compute_segment_length(model, cylinders, q, origin_muscle, insertion_muscle, plot_limit, plot_cadran)  
               
            else :  
               segment_length, _ = compute_segment_length(model, cylinders, q, origin_muscle, insertion_muscle, plot_all, plot_cadran)  
            
            # compute others y
            qs.append(qi)
            segment_lengths.append(segment_length)
            dlmt_dq = compute_dlmt_dq(model, q, cylinders, muscle_index, delta_qi = 1e-8)
            muscle_force = compute_fm_muscle_index(model, muscle_index, q, qdot, alpha)
            torque, _ = compute_torque(dlmt_dq, muscle_force)
            
            # add line
            writer.add_line(muscle_index, q, qdot, alpha, origin_muscle, insertion_muscle, segment_length, copy.deepcopy(dlmt_dq), muscle_force, torque)
         
         writer.close()
   

def plot_all_q_variation(model, q_fixed, y_label, filename="", file_path="") :
   
   """Create and save a plot png, y as a Function of q Values
   Examples for y : 'segment_length', 'muscle_force', 'torque'

   INPUTS
   - model : model 
   - q_fixed : array 4*1, q fixed, reference
   - y : string : colomn selected for the plot (y axis)
   - filename : string, name of the file to create
   - file_path : path for the directory with all file excel q
   """
   print(f"plot_all_q_variation : {y_label}")
   
   # Create a folder for save excel files and plots
   directory = file_path+"/plot_all_q_variation_" + filename
   create_directory(directory)
   
   if os.path.exists(f"{directory}/{y_label}_all_q_variation.png") == False :
   
      _, q_ranges_names_with_dofs = compute_q_ranges(model)
      
      row_fixed, col_fixed = compute_row_col(model.nbQ(), 3)
      fig, axs = plt.subplots(row_fixed, col_fixed, figsize=(15, 10))

      for q_index in range (model.nbQ()) : 
         file_q_index = f"{directory}/{q_index}_{q_ranges_names_with_dofs[q_index]}_" + filename+".xlsx"
         
         y = []
         qs = []
         df = pd.read_excel(file_q_index)

         # selected_columns = [f'q_{q_ranges_names_with_dofs[q_index]}']
         # y = df.loc[:, selected_columns].values
         qs.append(df.iloc[1:, q_index + 1].values)
         y.append(df.loc[1:, y_label].values)
         
         discontinuities = find_discontinuty(qs[0], y[0], plot_discontinuities=False)
         
         row = q_index // 3
         col = q_index % 3

         axs[row, col].plot(qs, y, marker='o', linestyle='-', color='b', markersize=3)
         for idx in discontinuities:
            axs[row, col].plot(qs[idx:idx+2], y[idx:idx+2], 'r', linewidth=2)  # Discontinuities are in red
         axs[row, col].set_xlabel(f'q{q_index} Variation',fontsize='smaller')
         axs[row, col].set_ylabel(y_label,fontsize='smaller')
         axs[row, col].set_title(f'{q_ranges_names_with_dofs[q_index]}',fontsize='smaller')
      
      fig.suptitle(f'{y_label} as a Function of q Values\nq_fixed = {q_fixed}', fontweight='bold')
      plt.tight_layout()  
      create_and_save_plot(f"{directory}", f"{y_label}_all_q_variation.png")
      plt.show()
   
   return None 

def data_for_learning_without_discontinuites_ddl(muscle_selected, cylinders, model, dataset_size, filename, num_points = 50, plot_cylinder_3D=False, plot_discontinuities = False, plot_cadran = False, plot_graph = False) :
   
   """
   Create a data frame for prepare datas without any discontinuities or error wrapping 
   Generate a mvt and then, remove problematic datas
   
   This function is used in "data_generation_muscles" but you can use it if you want to generate datas for one muscle
   Please, paid attention of "filename" you selected ... 
   You can not re generate datas for one muscle if files already exist, you need to delete files before !
   
   You can also use this function to add more lines (choose a bigger "dataset_size" value)
   If you choose a smaller "dataset_size" value, nothing will happen 
   
   INPUTS
   - muscle_selected : string, name of the muscle selected. 
                        Please chose an autorized name in this list : 
                        ['PECM2', 'PECM3', 'LAT', 'DELT2', 'DELT3', 'INFSP', 'SUPSP', 'SUBSC', 'TMIN', 'TMAJ',
                        'CORB', 'TRIlong', 'PECM1', 'DELT1', 'BIClong', 'BICshort']
   - cylinders : List of muscle's cylinder (0, 1 or 2 cylinders)
   - model : model 
   - dataset_size : int, number of data we would like
   - filename : string, name of the file to create
   - num_points : int (default = 50) number of point to generate per mvt
   - plot : bool (default false), True if we want a plot of point P, S (and Q, G, H and T) with cylinder(s)
   - plot_discontinuities : bool (default = False), true to show mvt with discontinuity
   - plot_cradran : bool (default = False), True to show cadran, pov of each cylinder and wrapping"""
   
   q_ranges, q_ranges_names_with_dofs = compute_q_ranges(model)
   muscle_index = find_index_muscle(model, muscle_selected)
   initialisation_generation(model, muscle_index, cylinders)
   
   writer = ExcelBatchWriter(filename+f"/{cylinders[0].muscle}.xlsx", q_ranges_names_with_dofs, batch_size=100)
   writer_datas_ignored = ExcelBatchWriter(filename+f"/{cylinders[0].muscle}_datas_ignored.xlsx", q_ranges_names_with_dofs, batch_size=100)
 
   # Limits of q
   min_vals_q = [row[0] for row in q_ranges]
   max_vals_q = [row[1] for row in q_ranges] 

   num_line = writer.get_num_line()
   while num_line < dataset_size : 

      # Generate a random q 
      q_ref = np.random.uniform(low=min_vals_q, high=max_vals_q)
      
      i = random.randint(0, model.nbQ() - 1)
      
      qs = []
      segment_lengths = []
      datas_ignored = []
      f_sup_limits = []
      lines = []
      to_remove = []
      
      q = copy.deepcopy(q_ref)
      
      # Generate points (num_points) of a mvt relatively to qi
      for k in range (num_points) : 
         
         # q, qdot, alpha
         qi = k * ((q_ranges[i][1] - q_ranges[i][0]) / num_points) + q_ranges[i][0]
         q[i] = qi
         qdot = np.array([random.uniform(-10*np.pi, 10*np.pi) for _ in range (len(q_ranges))])
         alpha = random.uniform(0, 1) 

         # compute lmt, fm, tau
         origin_muscle, insertion_muscle = update_points_position(model, [0, -1], muscle_index, q)
         
         segment_length, data_ignored = compute_segment_length(model, cylinders, q, origin_muscle, insertion_muscle, 
                                                               plot_cylinder_3D, plot_cadran)  
         dlmt_dq = compute_dlmt_dq(model, q, cylinders, muscle_index, delta_qi = 1e-8)
         muscle_force = compute_fm_muscle_index(model, muscle_index, q, qdot, alpha)
         torque, f_sup_limit = compute_torque(dlmt_dq, muscle_force)
         
         # keep informations to verify after if datas are correct
         qs.append(qi)
         segment_lengths.append(segment_length)
         datas_ignored.append(data_ignored)
         f_sup_limits.append(f_sup_limit)
         
         # add the new line
         lines.append([muscle_index, copy.deepcopy(q), copy.deepcopy(qdot), alpha, origin_muscle, insertion_muscle, segment_length, 
                       copy.deepcopy(dlmt_dq), muscle_force, torque])
      
      # Verify if datas are correct, no discontinuties, no errors
      # Find indexes with discontinuties
      discontinuities = find_discontinuty(qs, segment_lengths, plot_discontinuities = plot_discontinuities)
      for discontinuity in discontinuities : 
         min, max = data_to_remove_part(discontinuity, qs, num_points, 3)
         to_remove.extend(range(min, max + 1))
      # Find error wrapping
      positions = [n for n, ignored in enumerate(datas_ignored) if ignored]
      if len(positions) != 0 : 
         min, max = find_discontinuities_from_error_wrapping_range(positions, num_points, range = 3)
         to_remove.extend(range(min, max + 1))
      # Find error f_sup_limit 
      positions_f_sup_limit = [n for n, ignored in enumerate(f_sup_limits) if ignored]
      if len(positions_f_sup_limit) != 0 : 
         min, max = find_discontinuities_from_error_wrapping_range(positions_f_sup_limit, num_points, range = 0)
         to_remove.extend(range(min, max + 1))
      
      # Sort to keep only one occurancy of each indexes
      to_remove = sorted(set(to_remove), reverse=True)
      
      if len(to_remove) != 0 and plot_graph : 
         # To check points removed (in red)
         plot_mvt_discontinuities_in_red(i, qs, segment_lengths, to_remove)
      
      # add lines in excel file
      for l_idx in range (len(lines)) :
         if l_idx in to_remove : 
            writer_datas_ignored.add_line(*lines[l_idx])
         else : 
            writer.add_line(*lines[l_idx])
            num_line+=1
            if num_line > dataset_size : 
               break

   writer.close()
   writer_datas_ignored.close()
   
   if dataset_size != 0 :
      writer.del_lines(dataset_size)
   return None

def data_generation_muscles(muscles_selected, cylinders, model, dataset_size, dataset_size_noise, filename, num_points = 50, plot_cylinder_3D=False, plot_discontinuities = False, plot_cadran = False, plot_graph = False ) : 
   
   """Generate datas for all muscles selected, one file of "dataset_size" lines per muscle
   
   INPUTS
   - muscles_selected : [string], names of muscles selected. 
                        Please chose autorized names in this list : 
                        ['PECM2', 'PECM3', 'LAT', 'DELT2', 'DELT3', 'INFSP', 'SUPSP', 'SUBSC', 'TMIN', 'TMAJ',
                        'CORB', 'TRIlong', 'PECM1', 'DELT1', 'BIClong', 'BICshort']
   - cylinders : List of list of muscle's cylinder (0, 1 or 2 cylinders)
   - model : model 
   - dataset_size : int, number of data we would like, choose "0" to don't add more datas (pure)
   - dataset_size_noise : int, number of data with noise we would like, choose "0" to don't add more datas (noise)
   - filename : string, name of the file to create
   - num_points : int (default = 50) number of point to generate per mvt
   - plot : bool (default false), True if we want a plot of point P, S (and Q, G, H and T) with cylinder(s)
   - plot_discontinuities : bool (default = False), true to show mvt with discontinuity
   - plot_cradran : bool (default = False), True to show cadran, pov of each cylinder and wrapping
   
   """
   # Create a folder for save excel files and plots
   directory = "data_generation_" + filename
   create_directory(directory)
   
   for k in range(len(muscles_selected)) : 
      create_directory(f"{directory}/{muscles_selected[k]}")
      if dataset_size != 0 :
         data_for_learning_without_discontinuites_ddl(muscles_selected[k], cylinders[k], model, dataset_size, 
                                                   f"{directory}/{cylinders[k][0].muscle}", num_points, 
                                                   plot_cylinder_3D, plot_discontinuities, plot_cadran, plot_graph)
      
      if dataset_size_noise != 0 :
         data_for_learning_with_noise(f"{directory}/{cylinders[k][0].muscle}/{cylinders[0].muscle}.xlsx", dataset_size_noise)
      
      # Plot visualization
      q_fixed = np.array([0.0 for _ in range (model.nbQ())])
      
      create_all_q_variation_files(muscles_selected[k], cylinders[k], model, q_fixed, "", num_points = 100, 
                     plot_all = False, plot_limit = False, plot_cadran=False, file_path=f"{directory}/{cylinders[k][0].muscle}")
      plot_all_q_variation(model, q_fixed, 'segment_length', "", file_path=f"{directory}/{cylinders[k][0].muscle}")
      plot_all_q_variation(model, q_fixed, 'muscle_force', "", file_path=f"{directory}/{cylinders[k][0].muscle}")
      plot_all_q_variation(model, q_fixed, 'torque', "", file_path=f"{directory}/{cylinders[k][0].muscle}")
      
      plot_length_jacobian(model, q_fixed, cylinders[k], muscles_selected[k], f"{directory}/{cylinders[k][0].muscle}/plot_all_q_variation_", 100)
      print("stop")

def data_for_learning_with_noise(model, excel_file_path, dataset_size_noise, batch_size = 1000, noise_std_dev = 0.01) :
   """
   Create datas with noise un add on file _with_noise.xlsx
   
   INPUTS : 
   - model : biorbd model
   - excel_file_path : string, path of the original file with all pure datas (= without noise and not ignored)
   - dataset_size_noise : int, num of lines to have in the file
   - batch_size : (default = 1000) int, size of batch
      PLEASE, choose an appropriate value, len(df) must be a multiple of batch_size to add the correct num of row 
   - noise_std-dev : (default = 0.01) float, standard deviation of added noise 
   
   OUTPUT : 
   None, create or complete a file [...]_with_noise.xlsx
   """
   _, q_ranges_names_with_dofs = compute_q_ranges(model)
   
   writer = ExcelBatchWriterWithNoise(f"{excel_file_path.replace(".xlsx", "")}_with_noise.xlsx", q_ranges_names_with_dofs,
                                      batch_size, noise_std_dev)
   writer.augment_data_with_noise_batch(dataset_size_noise)