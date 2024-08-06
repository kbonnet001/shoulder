import numpy as np
import biorbd
from neural_networks.file_directory_operations import *

def compute_muscle_force_origin_insertion_nul(muscle_index, lmt, model_one_muscle = biorbd.Model("models/oneMuscle.bioMod")) :
    """
    Compute muscle force with lmt
    This is a temporarily function !
    
    We suppose that origin point = 0, 0, 0 and insertion point = 0, 0, 0
    Then, insertion point = 0, 0, lmt
    Please, paid attention to the file 'oneMuscle.bioMod'
    For the moment, there is only PECM2 and PECM3 with modified origin and insertion points
    
    """
    q = np.array([0])
    qdot = np.array([0])
    
    mus = model_one_muscle.muscle(muscle_index) 
    mus.position().setInsertionInLocal(np.array([0, 0, lmt])) 
    
    states = model_one_muscle.stateSet()
    for state in states:
        state.setActivation(1) # 1 ==> 100% activation
    f = model_one_muscle.muscleForces(states, q, qdot).to_array()

    print(f"f: {f[muscle_index]}")
    if f[muscle_index] >= 5000 : 
        print("ERROR : force >= 5000 !!!")
    
    return f[muscle_index]
    

def compute_torque(dlmt_dq, f) : 
    print("dlmt_dq = ", dlmt_dq)
    print("f = ", f)
    # torque = []
    # for i in range(len(f)) : 
    #     torque.append(sum(np.dot(- np.transpose(dlmt_dq[i]), f[i])))
    return sum(np.dot(- np.transpose(dlmt_dq), f))

def compute_torque_from_lmt_and_dlmt_dq(muscle_index, lmt, dlmt_dq) : 
    model_one_muscle = biorbd.Model("models/oneMuscle.bioMod")
    f = compute_muscle_force_origin_insertion_nul(muscle_index, lmt, model_one_muscle)
    return compute_torque(dlmt_dq, f)


# def plot_muscle_force_and_torque_q_variation(muscle_selected, cylinders, model, q_fixed, directory_path, num_points = 100) :
   
#     """Create a directory with all excel files and png of mvt for all q

#     INPUTS
#     - muscle_selected : string, name of the muscle selected. 
#                             Please chose an autorized name in this list : 
#                             ['PECM2', 'PECM3', 'LAT', 'DELT2', 'DELT3', 'INFSP', 'SUPSP', 'SUBSC', 'TMIN', 'TMAJ',
#                             'CORB', 'TRIlong', 'PECM1', 'DELT1', 'BIClong', 'BICshort']
#     - cylinders : List of muscle's cylinder (0, 1 or 2 cylinders)
#     - model : model 
#     - q_fixed : array 4*1, q fixed, reference
#     - filename : string, name of the file to create
#     - num_points : int (default = 50) number of point to generate per mvt
#     - plot_all : bool (default false), True if we want all plots of point P, S (and Q, G, H and T) with cylinder(s)
#     - plot_limit : bool (default = False), True to plot points P, S (and Q, G, H and T) with cylinder(s) 
#                                                                                             (first, middle and last one)
#     - plot_cradran : bool (default = False), True to show cadran, pov of each cylinder and wrapping"""

#     # Create a folder for save excel files and plots

#     q_ranges, q_ranges_names_with_dofs = compute_q_ranges(model)
#     muscle_index= find_index_muscle(model, muscle_selected)
#     q = copy.deepcopy(q_fixed)

#     row_fixed, col_fixed = compute_row_col(len(q_ranges), 3)
#     fig1, axs1 = plt.subplots(row_fixed, col_fixed, figsize=(15, 10))
#     fig2, axs2 = plt.subplots(row_fixed, col_fixed, figsize=(15, 10))

#     for q_index in range (len(q_ranges)) : 
#         forces = []
#         torques = []
#         qs = []
        
#         q = copy.deepcopy(q_fixed)
        
#         for k in range (num_points+1) : 
#             print("plot muscle force and torque, k = ", k)
            
#             qi = k * ((q_ranges[q_index][1] - q_ranges[q_index][0]) / num_points) + q_ranges[q_index][0]
#             q[q_index] = qi
            
#             print("q = ", q)

#             # ------------------------------------------------
#             origin_muscle, insertion_muscle = update_points_position(model, [0, -1], muscle_index, q)
#             lmt, _ = compute_segment_length(model, cylinders, muscle_index, q_ranges, q, origin_muscle, insertion_muscle, plot = False, plot_cadran = False)  
         
            
#             dlmt_dq = compute_dlmt_dq(model, q_ranges, q, cylinders, muscle_index, delta_qi = 1e-8)
#             muscle_force = compute_muscle_force_origin_insertion_nul(muscle_index, lmt)
#             torque = compute_torque(dlmt_dq, muscle_force)

#             qs.append(qi)
#             forces.append(muscle_force)
#             torques.append(torque)
        
#         row = q_index // 3
#         col = q_index % 3

#         axs1[row, col].plot(qs, forces, marker='o', linestyle='-', color='b', markersize=3)
#         axs1[row, col].set_xlabel(f'q{q_index} Variation',fontsize='smaller')
#         axs1[row, col].set_ylabel('Muscle_force',fontsize='smaller')
#         axs1[row, col].set_title(f'{q_ranges_names_with_dofs[q_index]}',fontsize='smaller')
#         axs1[row, col].set_xticks(qs[::5])
#         axs1[row, col].set_xticklabels([f'{x:.4f}' for x in qs[::5]],fontsize='smaller')
        
#         axs2[row, col].plot(qs, torques, marker='o', linestyle='-', color='b', markersize=3)
#         axs2[row, col].set_xlabel(f'q{q_index} Variation',fontsize='smaller')
#         axs2[row, col].set_ylabel('Torque',fontsize='smaller')
#         axs2[row, col].set_title(f'{q_ranges_names_with_dofs[q_index]}',fontsize='smaller')
#         axs2[row, col].set_xticks(qs[::5])
#         axs2[row, col].set_xticklabels([f'{x:.4f}' for x in qs[::5]],fontsize='smaller')

#     fig1.suptitle(f'Muscle Force as a Function of q Values\nq_fixed = {q_fixed}', fontweight='bold')
#     plt.tight_layout()  
#     create_and_save_plot(f"{directory_path}", "muscle_force_q_variation.png")
#     plt.show()
    
#     fig2.suptitle(f'Torque as a Function of q Values\nq_fixed = {q_fixed}', fontweight='bold')
#     plt.tight_layout()  
#     create_and_save_plot(f"{directory_path}", "torque_q_variation.png")
#     plt.show()

#     return None