import numpy as np
import matplotlib.pyplot as plt
from neural_networks.CSVBatchWriter import CSVBatchWriter
from neural_networks.discontinuities import *
from neural_networks.functions_data_generation import *
from neural_networks.file_directory_operations import create_directory, create_and_save_plot
import copy
import random
from neural_networks.muscles_length_jacobian import compute_dlmt_dq, plot_length_jacobian
from neural_networks.muscle_forces_and_torque import compute_fm_and_torque
import os
from neural_networks.muscle_plotting_utils import compute_row_col, plot_mvt_discontinuities_in_red
from neural_networks.CSVBatchWriterWithNoise import CSVBatchWriterWithNoise
import pandas as pd
from itertools import product

def test_limit_data_for_learning(muscle_selected, cylinders, model, plot=True, plot_cadran=False):
    """
    Test the limits of q values for the selected muscle.
    This function helps to observe extreme configurations for the muscle.

    Args:
    - muscle_selected : str, name of the selected muscle. Choose from the authorized list:
                        ['PECM2', 'PECM3', 'LAT', 'DELT2', 'DELT3', 'INFSP', 'SUPSP', 'SUBSC', 'TMIN', 'TMAJ',
                        'CORB', 'TRIlong', 'PECM1', 'DELT1', 'BIClong', 'BICshort']
    - cylinders : list of int, list of muscle cylinders (0, 1, or 2 cylinders)
    - model : model object
    - q_ranges : array of shape (4, 2), q ranges for the selected muscle
    - plot : bool, default True, whether to plot points P, S (and Q, G, H, and T) with cylinders
    - plot_cadran : bool, default False, whether to show cadran, view of each cylinder, and wrapping
    """

    # Get the index of the selected muscle from the model
    muscle_index = find_index_muscle(model, muscle_selected)

    # Initialize the model for generation with the selected muscle and cylinders
    initialisation_generation(model, muscle_index, cylinders)

    # Number of q parameters in the model
    nb_q = model.nbQ()

    # Compute the q ranges for the model
    q_ranges, _ = compute_q_ranges(model)

    # Initialize test limits for each q parameter
    q_test_limite = [[0., 0., 0.] for _ in range(nb_q)]
    for k in range(nb_q):
        q_test_limite[k][0] = q_ranges[k][0]
        q_test_limite[k][1] = (q_ranges[k][0] + q_ranges[k][1]) / 2
        q_test_limite[k][2] = q_ranges[k][1]

    # Check that all elements of q_test_limite have the same size
    sizes = [len(q) for q in q_test_limite]
    assert all(size == sizes[0] for size in sizes), "Dimension sizes are inconsistent."

    # Iterate over all combinations of indices
    for indices in product(range(sizes[0]), repeat=nb_q):
        # Construct q values for the current combination of indices
        q = np.array([q_test_limite[dim][index] for dim, index in enumerate(indices)])
        print("Indices:", indices)
        print("q =", q)
        
        # Update the positions of the muscle origin and insertion points based on q
        origin_muscle, insertion_muscle = update_points_position(model, [0, -1], muscle_index, q)
        
        # Compute segment length and optionally plot results
        segment_length, _ = compute_segment_length(model, cylinders, q, origin_muscle, insertion_muscle, plot, 
                                                   plot_cadran)  
        print("segment_length =", segment_length)


def get_lines_to_remove(qs, segment_lengths, datas_ignored, f_sup_limits, num_points, plot_discontinuities):
    """
    Determine which data points to remove based on discontinuities, ignored data, and errors.

    Args:
        qs (list): List of data points to check for discontinuities.
        segment_lengths (list): Lengths of segments for finding discontinuities.
        datas_ignored (list): Boolean list indicating which data points are ignored.
        f_sup_limits (list): Boolean list indicating limits where errors occur.
        num_points (int): Total number of data points.
        plot_discontinuities (bool): Flag to plot discontinuities for visualization.

    Returns:
        list: List of indices to be removed.
    """
    
    # Initialize the list to store indices to be removed
    to_remove = []
    
    # Verify if data is correct and find discontinuities
    discontinuities = find_discontinuity(qs, segment_lengths, plot_discontinuities=plot_discontinuities)
    for discontinuity in discontinuities:
        min_idx, max_idx = data_to_remove_part(discontinuity, qs, num_points, range=3)
        to_remove.extend(range(min_idx, max_idx + 1))
    
    # Find indices with errors in ignored data
    positions = [n for n, ignored in enumerate(datas_ignored) if ignored]
    if positions:
        min_idx, max_idx = find_discontinuities_from_error_wrapping_range(positions, num_points, range=3)
        to_remove.extend(range(min_idx, max_idx + 1))
    
    # Find indices with errors in f_sup_limits
    positions_f_sup_limit = [n for n, ignored in enumerate(f_sup_limits) if ignored]
    if positions_f_sup_limit:
        min_idx, max_idx = find_discontinuities_from_error_wrapping_range(positions_f_sup_limit, num_points, range=0)
        to_remove.extend(range(min_idx, max_idx + 1))
    
    # Remove duplicate indices and sort in descending order
    to_remove = sorted(set(to_remove), reverse=True)
    
    return to_remove

def plot_one_q_variation(muscle_selected, cylinders, model, q_fixed, i, filename, num_points=100, plot_all=False, plot_limit=False, plot_cadran=False):
   """
   Create a directory with a CSV file and a PNG plot for varying one q parameter.

   Args:
   - muscle_selected : str, name of the selected muscle. Choose from the authorized list:
                     ['PECM2', 'PECM3', 'LAT', 'DELT2', 'DELT3', 'INFSP', 'SUPSP', 'SUBSC', 'TMIN', 'TMAJ',
                     'CORB', 'TRIlong', 'PECM1', 'DELT1', 'BIClong', 'BICshort']
   - cylinders : list of int, list of muscle cylinders (0, 1, or 2 cylinders)
   - model : model object
   - q_fixed : array of shape (4,), fixed q values (reference)
   - i : int, index of the q parameter to vary (0, 1, 2, or 3)
   - filename : str, name of the file to create
   - num_points : int, number of points to generate per movement (default 100)
   - plot_all : bool, if True, plot all points P, S (and Q, G, H, T) with cylinders (default False)
   - plot_limit : bool, if True, plot points P, S (and Q, G, H, T) with cylinders at first, middle, and last points (default False)
   - plot_cadran : bool, if True, show cadran, POV of each cylinder, and wrapping (default False)
   """

   # Compute q ranges and associated DOF names
   q_ranges, q_ranges_names_with_dofs = compute_q_ranges(model)

   # Get the index of the selected muscle
   muscle_index = find_index_muscle(model, muscle_selected)

   # Initialize the model with the selected muscle and cylinders
   initialisation_generation(model, muscle_index, cylinders)

   # Create a directory for saving the output
   directory = "plot_one_q_variation_" + filename
   create_directory(directory)

   # Initialize CSV writer for saving data
   writer = CSVBatchWriter(f"{directory}/{filename}.CSV", q_ranges_names_with_dofs, model.nbMuscles(), model.nbQ(), batch_size=100)

   # Initialize variables for storing results
   q = q_fixed
   segment_lengths = []
   qs = []

   # Loop through points to vary the selected q parameter
   for k in range(num_points + 1):
      print("Plotting one q variation, k =", k)
      
      # Calculate current q value
      qi = k * ((q_ranges[i][1] - q_ranges[i][0]) / num_points) + q_ranges[i][0]
      q[i] = qi
      
      # Default values for qdot and alpha
      qdot = np.zeros(model.nbQ())
      alpha = 1
      
      # Update muscle points' positions
      origin_muscle, insertion_muscle = update_points_position(model, [0, -1], muscle_index, q)
      
      # Compute segment length, conditionally plot limits if specified
      if k in [0, num_points / 2, num_points] and plot_limit:
         segment_length, _ = compute_segment_length(model, cylinders, q, origin_muscle, insertion_muscle, plot_limit, plot_cadran)
      else:
         segment_length, _ = compute_segment_length(model, cylinders, q, origin_muscle, insertion_muscle, plot_all, plot_cadran)
      
      # Append results for plotting
      qs.append(qi)
      segment_lengths.append(segment_length)
      
      # Compute muscle length Jacobian and other metrics
      dlmt_dq = model.musclesLengthJacobian(q).to_array()[muscle_index]
      # dlmt_dq = compute_dlmt_dq(model, q, cylinders, muscle_index, delta_qi = 1e-8) # calcul avec wrapping
      muscle_force, torque = compute_fm_and_torque(model, muscle_index, q, qdot, alpha)
      # muscle_force = compute_fm(model, q, qdot, alpha)[muscle_index]
      # torque, f_sup_limit = compute_torque(dlmt_dq, muscle_force)

      # Write data to CSV
      writer.add_line(muscle_index, q, qdot, alpha, origin_muscle, insertion_muscle, segment_length, copy.deepcopy(dlmt_dq), muscle_force, torque)

   # Plot the results
   plt.plot(qs, segment_lengths, marker='o', linestyle='-', color='b')
   plt.xlabel(f'q{i}')
   plt.ylabel('Muscle Length')
   plt.title(f'Muscle Length as a Function of q{i} Values')
   plt.xticks(qs[::5])
   plt.yticks(segment_lengths[::5])
   plt.grid(True)
   
   # Save and display the plot
   create_and_save_plot(f"{directory}", "one_q_variation.png")
   plt.show()
   
   # Analyze and plot any discontinuities in the data
   find_discontinuity(qs, segment_lengths, plot_discontinuities=True)
   
   # Close the CSV writer
   writer.close()

def create_all_q_variation_files(muscle_selected, cylinders, model, q_fixed, filename, num_points=100, plot_all=False, plot_limit=False, plot_cadran=False, file_path=""):
   """
   Create a directory with CSV files for all q parameters, varying one q at a time while keeping others fixed.

   Args:
   - muscle_selected : str, name of the selected muscle. Choose from the authorized list:
                     ['PECM2', 'PECM3', 'LAT', 'DELT2', 'DELT3', 'INFSP', 'SUPSP', 'SUBSC', 'TMIN', 'TMAJ',
                     'CORB', 'TRIlong', 'PECM1', 'DELT1', 'BIClong', 'BICshort']
   - cylinders : list of int, list of muscle cylinders (0, 1, or 2 cylinders)
   - model : model biorbd
   - q_fixed : array of shape (4,), fixed q values (reference)
   - filename : str, name of the file to create
   - num_points : int, number of points to generate per movement (default 100)
   - plot_all : bool, if True, plot all points P, S (and Q, G, H, T) with cylinders (default False)
   - plot_limit : bool, if True, plot points P, S (and Q, G, H, T) with cylinders at first, middle, and last points (default False)
   - plot_cadran : bool, if True, show cadran, POV of each cylinder, and wrapping (default False)
   - file_path : str, path for the directory where files will be saved (default is the current directory)
   """

   # Create a directory for saving CSV files and plots
   directory = f"{file_path}/plot_all_q_variation_{filename}"
   create_directory(directory)

   # Compute q ranges and associated DOF names
   q_ranges, q_ranges_names_with_dofs = compute_q_ranges(model)

   # Get the index of the selected muscle
   muscle_index = find_index_muscle(model, muscle_selected)

   # Initialize the model with the selected muscle and cylinders
   initialisation_generation(model, muscle_index, cylinders)

   # Copy the fixed q values
   q = copy.deepcopy(q_fixed)

   # Iterate over all q parameters
   for q_index in range(model.nbQ()):
      # Define the file path for the current q parameter
      file_path = f"{directory}/{q_index}_{q_ranges_names_with_dofs[q_index]}_{filename}.CSV"

      # Check if the CSV file already exists; if not, create a new writer
      if not os.path.exists(file_path):
         writer = CSVBatchWriter(file_path, q_ranges_names_with_dofs, model.nbMuscles(), model.nbQ(), batch_size=100)

         # Initialize lists to store results
         segment_lengths = []
         qs = []

         # Loop through points to vary the current q parameter
         for k in range(num_points + 1):
               print("Plotting all q variation, k =", k)

               # Calculate the current q value
               qi = k * ((q_ranges[q_index][1] - q_ranges[q_index][0]) / num_points) + q_ranges[q_index][0]
               q[q_index] = qi

               # Default values for qdot and alpha
               qdot = np.zeros(model.nbQ())
               alpha = 1

               # Update muscle points' positions
               origin_muscle, insertion_muscle = update_points_position(model, [0, -1], muscle_index, q)

               # Compute segment length, with conditional plotting if specified
               if k in [0, num_points / 2, num_points] and plot_limit:
                  segment_length, _ = compute_segment_length(model, cylinders, q, origin_muscle, insertion_muscle, plot_limit, plot_cadran)
               else:
                  segment_length, _ = compute_segment_length(model, cylinders, q, origin_muscle, insertion_muscle, plot_all, plot_cadran)

               # Append results for plotting
               qs.append(qi)
               segment_lengths.append(segment_length)

               # Compute muscle length Jacobian and other metrics
               dlmt_dq = model.musclesLengthJacobian(q).to_array() #[muscle_index]
               # dlmt_dq = compute_dlmt_dq(model, q, cylinders, muscle_index, delta_qi = 1e-8) # calcul avec wrapping
               muscle_force, torque = compute_fm_and_torque(model, muscle_index, q, qdot, alpha)
               # muscle_force = compute_fm(model, q, qdot, alpha)[muscle_index]
               # torque, f_sup_limit = compute_torque(dlmt_dq, muscle_force)

               # Write data to CSV
               writer.add_line(muscle_index, q, qdot, alpha, origin_muscle, insertion_muscle, segment_length, 
                               copy.deepcopy(dlmt_dq), copy.deepcopy(muscle_force), copy.deepcopy(torque))

         # Close the CSV writer
         writer.close()

def plot_all_q_variation(model, q_fixed, y_label, filename="", file_path=""):
   """
   Create and save a plot PNG of y as a function of q values for each q parameter.

   Args:
   - model : model object
   - q_fixed : array of shape (4,), fixed q values (reference)
   - y_label : str, name of the column to plot on the y-axis (e.g., 'segment_length', 'muscle_force', 'torque')
   - filename : str, name of the file to create (default is empty string)
   - file_path : str, path for the directory where CSV files and plots are saved (default is empty string)
   """
   print(f"Plotting all q variations: {y_label}")

   # Create a directory for saving CSV files and plots
   directory = f"{file_path}/plot_all_q_variation_{filename}"
   create_directory(directory)
   
   # Check if the plot file already exists
   if not os.path.exists(f"{directory}/{y_label}_all_q_variation.png"):
      # Compute q ranges and associated DOF names
      _, q_ranges_names_with_dofs = compute_q_ranges(model)
      
      # Compute the number of rows and columns for subplots
      row_fixed, col_fixed = compute_row_col(model.nbQ(), 3)
      fig, axs = plt.subplots(row_fixed, col_fixed, figsize=(15, 10))

      # Loop through each q parameter
      for q_index in range(model.nbQ()):
         # Define the file path for the current q parameter
         file_q_index = f"{directory}/{q_index}_{q_ranges_names_with_dofs[q_index]}_{filename}.CSV"
         
         # Initialize lists to store y values and q values
         y = []
         qs = []

         # Read the CSV file into a DataFrame
         df = pd.read_csv(file_q_index)

         # Extract the q values and y values from the DataFrame
         qs.append(df.iloc[1:, q_index + 1].values)
         y.append(df.loc[1:, y_label].values)
         
         # Find discontinuities in the data
         discontinuities = find_discontinuity(qs[0], y[0], plot_discontinuities=False)
         
         # Determine the subplot position
         row = q_index // 3
         col = q_index % 3

         # Plot the data for the current q parameter
         axs[row, col].plot(qs[0], y[0], marker='o', linestyle='-', color='b', markersize=3)
         for idx in discontinuities:
               axs[row, col].plot(qs[0][idx:idx+2], y[0][idx:idx+2], 'r', linewidth=2)  # Discontinuities are shown in red
         axs[row, col].set_xlabel(f'q{q_index} Variation', fontsize='smaller')
         axs[row, col].set_ylabel(y_label, fontsize='smaller')
         axs[row, col].set_title(f'{q_ranges_names_with_dofs[q_index]}', fontsize='smaller')
      
      # Set the main title for the figure
      fig.suptitle(f'{y_label} as a Function of q Values\nq_fixed = {q_fixed}', fontweight='bold')
      plt.tight_layout()  
      
      # Save and display the plot
      create_and_save_plot(f"{directory}", f"{y_label}_all_q_variation.png")
      plt.show()

def data_for_learning_without_discontinuites_ddl(muscle_selected, cylinders, model, dataset_size, filename, num_points = 50, plot_cylinder_3D=False, plot_discontinuities = False, plot_cadran = False, plot_graph = False) :
   
   """
   Create a data frame for prepare datas without any discontinuities or error wrapping 
   Generate a mvt and then, remove problematic datas
   
   This function is used in "data_generation_muscles" but you can use it if you want to generate datas for one muscle
   Please, paid attention of "filename" you selected ... 
   You can not re generate datas for one muscle if files already exist, you need to delete files before !
   
   You can also use this function to add more lines (choose a bigger "dataset_size" value)
   If you choose a smaller "dataset_size" value, nothing will happen 
   
   Args
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
   
   # Create to writer to save datas, one with "purs datas" (no errors, discontinuties, etc) and an other with errors
   # To see distribution of datas (please, choose plot_discontinuities = True)
   writer = CSVBatchWriter(filename+f"/{cylinders[0].muscle}.CSV", q_ranges_names_with_dofs, model.nbMuscles(), model.nbQ(), batch_size=100)
   writer_datas_ignored = CSVBatchWriter(filename+f"/{cylinders[0].muscle}_datas_ignored.CSV", q_ranges_names_with_dofs, model.nbMuscles(), model.nbQ(), batch_size=100)
 
   # Limits of q
   min_vals_q = [row[0] for row in q_ranges]
   max_vals_q = [row[1] for row in q_ranges] 

   num_line = writer.get_num_line() 
   while num_line < dataset_size : 

      # Generate a random q 
      q_ref = np.random.uniform(low=min_vals_q, high=max_vals_q)
      i = random.randint(0, model.nbQ() - 1)
      
      # to verify errors and dicontinuities
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
         
         # Append results for plotting
         qs.append(qi)
         segment_lengths.append(segment_length)
         datas_ignored.append(data_ignored)
         # f_sup_limits.append(f_sup_limit)

         # Compute muscle length Jacobian and other metrics
         dlmt_dq = model.musclesLengthJacobian(q).to_array()
         # dlmt_dq = compute_dlmt_dq(model, q, cylinders, muscle_index, delta_qi = 1e-8) # calcul avec wrapping
         muscle_force, torque = compute_fm_and_torque(model, muscle_index, q, qdot, alpha)
         # muscle_force = compute_fm(model, q, qdot, alpha)[muscle_index]
         # torque, f_sup_limit = compute_torque(dlmt_dq, muscle_force)
         
         # add the new line
         lines.append([muscle_index, copy.deepcopy(q), copy.deepcopy(qdot), alpha, origin_muscle, insertion_muscle, segment_length, 
                       copy.deepcopy(dlmt_dq), muscle_force, torque])
      
      # Verify if datas are correct, no discontinuties, no errors
      # Find indexes with discontinuties
      to_remove = get_lines_to_remove(qs, segment_lengths, datas_ignored, f_sup_limits, num_points, plot_discontinuities)
      
      if plot_graph and len(to_remove) != 0 : 
         # To check points removed (in red)
         plot_mvt_discontinuities_in_red(i, qs, segment_lengths, to_remove)
      
      # add lines in csv file
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
   
   Args
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
   # Create a folder for save csv files and plots
   directory = "data_generation_" + filename
   create_directory(directory)
   
   for k in range(len(muscles_selected)) : 
      # Create a folder for muscle selected
      create_directory(f"{directory}/{muscles_selected[k]}")
      if dataset_size != 0 :
         data_for_learning_without_discontinuites_ddl(muscles_selected[k], cylinders[k], model, dataset_size, 
                                                   f"{directory}/{cylinders[k][0].muscle}", num_points, 
                                                   plot_cylinder_3D, plot_discontinuities, plot_cadran, plot_graph)
      
      if dataset_size_noise != 0 :
         data_for_learning_with_noise(f"{directory}/{cylinders[k][0].muscle}/{cylinders[0].muscle}.CSV", dataset_size_noise)
      
      # Plot visualization
      # q_fixed = np.array([0.0 for _ in range (model.nbQ())])
      q_fixed  = np.array([0.0,0.0,0.0,0.0,0.0,0.0,-1.4311,0.0]) # T pose
      
      create_all_q_variation_files(muscles_selected[k], cylinders[k], model, q_fixed, "", num_points = 100, 
                     plot_all = False, plot_limit = False, plot_cadran=False, file_path=f"{directory}/{cylinders[k][0].muscle}")
      plot_all_q_variation(model, q_fixed, 'segment_length', "", file_path=f"{directory}/{cylinders[k][0].muscle}")
      plot_all_q_variation(model, q_fixed, 'muscle_force', "", file_path=f"{directory}/{cylinders[k][0].muscle}")
      plot_all_q_variation(model, q_fixed, 'torque', "", file_path=f"{directory}/{cylinders[k][0].muscle}")
      
      plot_length_jacobian(model, q_fixed, cylinders[k], muscles_selected[k], f"{directory}/{cylinders[k][0].muscle}/plot_all_q_variation_", 100)
      print("stop")

def data_for_learning_with_noise(model, csv_file_path, dataset_size_noise, batch_size = 1000, noise_std_dev = 0.01) :
   """
   Create datas with noise un add on file _with_noise.CSV
   
   Args : 
   - model : biorbd model
   - csv_file_path : string, path of the original file with all pure datas (= without noise and not ignored)
   - dataset_size_noise : int, num of lines to have in the file
   - batch_size : (default = 1000) int, size of batch
      PLEASE, choose an appropriate value, len(df) must be a multiple of batch_size to add the correct num of row 
   - noise_std-dev : (default = 0.01) float, standard deviation of added noise 
   
   Returns : 
   None, create or complete a file [...]_with_noise.CSV
   """
   _, q_ranges_names_with_dofs = compute_q_ranges(model)
   
   writer = CSVBatchWriterWithNoise(f"{csv_file_path.replace(".CSV", "")}_with_noise.CSV", q_ranges_names_with_dofs, model.nbMuscles(), model.nbQ(),
                                      batch_size, noise_std_dev)
   writer.augment_data_with_noise_batch(dataset_size_noise)


