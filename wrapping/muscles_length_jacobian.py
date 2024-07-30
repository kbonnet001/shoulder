from scipy.linalg import norm
from neural_networks.functions_data_generation import update_points_position
import copy
import matplotlib.pyplot as plt
from neural_networks.functions_data_generation import *
import numpy as np
from neural_networks.other import compute_row_col
import os
from neural_networks.file_directory_operations import create_directory, create_and_save_plot

# -------------------------------------
def dev_partielle_lmt_qi_points_without_wrapping (lmt1, lmt2, delta_qi) : 
    
   """ Compute the partial derivative of lmt as a function of qi
   
   INPUTS : 
   - lmt1 : float, lmt1 = lmt(q + vect_delta_qi)
   - lmt2 : float, lmt2 = lmt(q - vect_delta_qi)
   - delta_qi : float, small variation of  q (for example, 1e-3, 1e-4, ...)
   
   OUTPUT :
   - dlmt_dqi : float, partial derivative of lmt as a function of qi   """
   
   return (lmt1 - lmt2) / (2*delta_qi)

def compute_dlmt_dq(model, q_ranges, q, cylinders, muscle_index, delta_qi=1e-8) : 
    
   """ Compute the partial derivative of lmt as a function of q 
   
   NOTE : We can't use directly "compute_segment_length" because it just return le length in regard of wrapping
   But in this case, if there is no wrapping, we compute dlmt_dq thank via points. 
   None alternatives exist for a situation "wrapping + via points" 
   so the resultats with biorbd could be very different (discontnuities)
   
   INPUTS : 
   - model : biorbd.Model
   - q : array qi 
   - cylinders : list of Cylinder, cylinders associate with muscle selected (=[] if no cylinders)
   - muscle_index : index of muscle selected
   - delta_qi : (default = 1e-8) float, a small variation for all qi 
   
   OUTPUS : 
   - dlmt_dq : array 1*len(q) float, the partial derivative of lmt as a function of q """
   
   dlmt_dq = []
   initialisation_generation(model, q_ranges, muscle_index, cylinders)
   
   for i in range(len(q)) : 
        
      # Create vect delta_qi
      vect_delta_qi = [0 for k in range(len(q))]
      vect_delta_qi[i] = delta_qi
      
      if len(cylinders) != 0 : 
        # if no cylinders, use via points to compute le length
        origin_point_pos, insertion_point_pos = update_points_position(model, [0, -1], muscle_index, q + vect_delta_qi)
        origin_point_neg, insertion_point_neg = update_points_position(model, [0, -1], muscle_index, q - vect_delta_qi)
          
        lmt1, _ = compute_segment_length(model, cylinders, q, origin_point_pos, insertion_point_pos, False, False)
        lmt2, _ = compute_segment_length(model, cylinders, q, origin_point_neg, insertion_point_neg, False, False)
      else : 
        mus = model.muscle(muscle_index) 
        p_pos = list(update_points_position(model, [n for n in range(len(mus.musclesPointsInGlobal(model, q)))], muscle_index, q + vect_delta_qi))
        p_neg = list(update_points_position(model, [n for n in range(len(mus.musclesPointsInGlobal(model, q)))], muscle_index, q - vect_delta_qi))
        
        lmt1 = sum(norm(np.array(p_pos[n]) - np.array(p_pos[n+1])) for n in range (len(p_pos)-1))
        lmt2 = sum(norm(np.array(p_neg[n]) - np.array(p_neg[n+1])) for n in range (len(p_neg)-1))
   
      dlmt_dqi = dev_partielle_lmt_qi_points_without_wrapping(lmt1, lmt2, delta_qi)
      
      dlmt_dq.append(copy.deepcopy(dlmt_dqi))
      
   print("dlmt_dq = ", dlmt_dq)
   
   return dlmt_dq
 
 
def plot_all_length_jacobian(model, q_fixed, cylinders, muscle_selected, filename, num_points = 100) : 
    
   """ Plot and save a comparaison of variation of dlmt_dq (with wrappings) and dlmt_dq_biorbd (with via points) 
   NOTE : you must have len(q_fixed) >=2
   
   INPUTS : 
   - model : biorbd.Model
   - q_fixed : array q value choosen by user
   - cylinders : list of Cylinder, cylinders associate with muscle selected (=[] if no cylinders)
   - muscle_selected : string, muscle selected
   - filename : string, filename/file_path to save the plot
   - num_points : (default = 100) int, number of points for plot
   
   OUTPUT : 
   - None, save plot"""
   
   print("plot_all_length_jacobian")
   
   if os.path.exists(f"{filename}/dlmt_dq.png") == False :
   
      q_ranges, q_ranges_names_with_dofs = compute_q_ranges(model)
      
      muscle_index= find_index_muscle(model, muscle_selected)
      delta_qi = 1e-10
      
      row_size, col_size = compute_row_col(len(q_ranges), 3)
      # Big fig with all dlmt_dq
      fig, axs = plt.subplots(row_size, col_size, figsize=(15, 10))
      
      # Compute the partial derivative of lmt as a function of q_index
      for q_index in range (len(q_ranges)) : 
         q = copy.deepcopy(q_fixed)
         qs = []
         dlmt_dqis= []
         dlmt_dq_biorbds = []
         
         # Then, q_index variate beetween min_range(q_index) and max_range(q_index)
         # For each value of q_index, compute dlmt_dq_index. So, we obtain an array of len(q_ranges) values
         # So, each subplot represent the varaition of dlmt_dq_index for each qi
         for k in range (num_points+1) : 
            print("q_index = ", q_index, " ; k = ", k)

            qi = k * ((q_ranges[q_index][1] - q_ranges[q_index][0]) / num_points) + q_ranges[q_index][0]
            q[q_index] = qi
            
            model.updateMuscles(q) 
            model.UpdateKinematicsCustom(q)
            dlmt_dq_biorbd = model.musclesLengthJacobian().to_array()
            
            dlmt_dq = compute_dlmt_dq(model, q_ranges, q, cylinders, muscle_index, delta_qi)
            
            qs.append(qi)
            dlmt_dqis.append(dlmt_dq)
            dlmt_dq_biorbds.append(dlmt_dq_biorbd[muscle_index])
         

         row = q_index // 3
         col = q_index % 3

         for i in range(len(dlmt_dq)) :
            if i == 0:
               axs[row, col].plot(qs, [dlmt_dqi_biorbd[i] for dlmt_dqi_biorbd in dlmt_dq_biorbds], marker='^', 
                                 linestyle='--', color = "silver", markersize=2, label=f"dlmt_dq_biorbd")
            else:
               axs[row, col].plot(qs, [dlmt_dqi_biorbd[i] for dlmt_dqi_biorbd in dlmt_dq_biorbds], marker='^',
                                 linestyle='--', color = "silver", markersize=2)
               
            axs[row, col].plot(qs, [dlmt_dqi[i] for dlmt_dqi in dlmt_dqis], marker='o', 
                              linestyle='-', markersize=2, label=f"dlmt_dq{i}")
         axs[row, col].set_xlabel(f'q{q_index} Variation',fontsize='smaller')
         axs[row, col].set_ylabel(f'dlmt_dq{q_index}',fontsize='smaller')
         axs[row, col].set_title(f'{q_ranges_names_with_dofs[q_index]}',fontsize='smaller')
         axs[row, col].legend()
      
      fig.suptitle(f'dlmt_dq\nq_fixed = {q_fixed}', fontweight='bold')
      plt.tight_layout()  
      create_and_save_plot(f"{filename}", "dlmt_dq.png")
      plt.show()


def plot_one_length_jacobian(model, q_fixed, cylinders, muscle_selected, filename, num_points = 100) : 
    
   """ Plot and save a comparaison of variation of dlmt_dq (with wrappings) and dlmt_dq_biorbd (with via points) 
   NOTE : you must have len(q_fixed) >=2
   
   INPUTS : 
   - model : biorbd.Model
   - q_fixed : array q value choosen by user
   - cylinders : list of Cylinder, cylinders associate with muscle selected (=[] if no cylinders)
   - muscle_selected : string, muscle selected
   - filename : string, filename/file_path to save the plot
   - num_points : (default = 100) int, number of points for plot
   
   OUTPUT : 
   - None, save plot"""
   
   print("plot_one_length_jacobian")
   
   q_ranges, q_ranges_names_with_dofs = compute_q_ranges(model)
   
   muscle_index= find_index_muscle(model, muscle_selected)
   delta_qi = 1e-10
   
   row_size, col_size = compute_row_col(len(q_ranges), 3)
   
   # Compute the partial derivative of lmt as a function of q_index
   for q_index in range (len(q_ranges)) : 
      
      if os.path.exists(f"{filename}/dlmt_dq{q_index}.png") == False :
         q = copy.deepcopy(q_fixed)
         qs = []
         dlmt_dqis= []
         dlmt_dq_biorbds = []
         
         # Then, q_index variate beetween min_range(q_index) and max_range(q_index)
         # For each value of q_index, compute dlmt_dq_index. So, we obtain an array of len(q_ranges) values
         # So, each fig represent plots for one partial derivative of lmt as a function of q_index
         # And, each subplot represent the varaition of dlmt_dq_index for THIS q_index
         
         for k in range (num_points+1) : 
            print("q_index = ", q_index, " ; k = ", k)

            qi = k * ((q_ranges[q_index][1] - q_ranges[q_index][0]) / num_points) + q_ranges[q_index][0]
            q[q_index] = qi
            
            model.updateMuscles(q) 
            model.UpdateKinematicsCustom(q)
            dlmt_dq_biorbd = model.musclesLengthJacobian().to_array()
            
            dlmt_dq = compute_dlmt_dq(model, q_ranges, q, cylinders, muscle_index, delta_qi)
            
            qs.append(qi)
            dlmt_dqis.append(dlmt_dq)
            dlmt_dq_biorbds.append(dlmt_dq_biorbd[muscle_index])
         
         fig_qi, axs_qi = plt.subplots(row_size, col_size, figsize=(15, 10))
         for j in range (len(dlmt_dq)) : 
            row_qi = j // 3
            col_qi = j % 3
            
            acc = np.mean(abs( np.array([dlmt_dq_biorbd[j] for dlmt_dq_biorbd in dlmt_dq_biorbds]) - 
                              np.array([dlmt_dq[j] for dlmt_dq in dlmt_dqis])))
            
            axs_qi[row_qi, col_qi].plot(qs, [dlmt_dq_biorbd[j] for dlmt_dq_biorbd in dlmt_dq_biorbds], 
                                       marker='^', linestyle='--', color = "silver", 
                                       markersize=2,label=f"dlmt_dq_biorbd{q_index}[{j}]")
            axs_qi[row_qi, col_qi].plot(qs, [dlmt_dq[j] for dlmt_dq in dlmt_dqis], marker='o', linestyle='-', 
                                       markersize=2, label=f"dlmt_dq{q_index}[{j}]")
            axs_qi[row_qi, col_qi].set_xlabel(f'q{q_index} Variation',fontsize='smaller')
            axs_qi[row_qi, col_qi].set_ylabel(f'dlmt_dq{q_index}[{j}]',fontsize='smaller')
            axs_qi[row_qi, col_qi].set_title(f'{q_ranges_names_with_dofs[q_index]}[{j}] - acc = {acc:.6f}',fontsize='smaller')
            axs_qi[row_qi, col_qi].legend()
         
         fig_qi.suptitle(f'dlmt_dq{q_index}\nq_fixed = {q_fixed}', fontweight='bold')
         plt.tight_layout()  
         create_and_save_plot(f"{filename}", f"dlmt_dq{q_index}.png")
         plt.show()

def plot_length_jacobian(model, q_fixed, cylinders, muscle_selected, directory_name, num_points = 100) :
   directory = f"{directory_name}/plot_length_jacobian"
   create_directory(directory)
   
   plot_one_length_jacobian(model, q_fixed, cylinders, muscle_selected, directory, num_points)
   plot_all_length_jacobian(model, q_fixed, cylinders, muscle_selected, directory, num_points)