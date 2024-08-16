from scipy.linalg import norm
from neural_networks.functions_data_generation import update_points_position
import copy
import matplotlib.pyplot as plt
from neural_networks.functions_data_generation import *
import numpy as np
from neural_networks.muscle_plotting_utils import compute_row_col
import os
from neural_networks.file_directory_operations import create_directory, create_and_save_plot

def dev_partielle_lmt_qi_points_without_wrapping(lmt1, lmt2, delta_qi):
    """
    Compute the partial derivative of lmt with respect to qi.

    This function calculates the central difference approximation of the derivative, which is a numerical method
    used to estimate derivatives by evaluating the function at slightly shifted points around the point of interest.

    Args:
        lmt1 (float): The value of lmt at the point (q + delta_qi), i.e., lmt1 = lmt(q + vect_delta_qi).
        lmt2 (float): The value of lmt at the point (q - delta_qi), i.e., lmt2 = lmt(q - vect_delta_qi).
        delta_qi (float): A small variation in q (e.g., 1e-3, 1e-4) used for numerical differentiation.

    Returns:
        float: The estimated partial derivative of lmt with respect to qi.
    """

    # Calculate the central difference to approximate the derivative
    return (lmt1 - lmt2) / (2 * delta_qi)

def compute_dlmt_dq(model, q, cylinders, muscle_index, delta_qi=1e-8):
    """
    Compute the partial derivative of the muscle-tendon length (lmt) with respect to joint angles (q).

    This function estimates the derivative of lmt with respect to each component of q using numerical differentiation.
    The calculation considers both wrapping and via points. It is important to note that biorbd may produce
    different results due to discontinuities in situations involving both wrapping and via points.

    Args:
        model (biorbd.Model): The biomechanical model containing muscles and joints.
        q (array-like): The joint angles at which to compute the derivatives.
        cylinders (list): A list of Cylinder objects associated with the selected muscle (empty if no cylinders).
        muscle_index (int): The index of the muscle for which the derivative is computed.
        delta_qi (float, optional): A small variation for each q component used for numerical differentiation (default is 1e-8).

    Returns:
        list of float: An array of partial derivatives of lmt with respect to each component of q.
    """

    dlmt_dq = []
    initialisation_generation(model, muscle_index, cylinders)  # Initialize generation for computation

    # Iterate over each joint angle to compute its partial derivative
    for i in range(len(q)):
        # Create a vector for small variation in the ith joint angle
        vect_delta_qi = [0 for k in range(len(q))]
        vect_delta_qi[i] = delta_qi

        if len(cylinders) != 0:
            # If cylinders are present, use the update_points_position function for length calculation
            origin_point_pos, insertion_point_pos = update_points_position(model, [0, -1], muscle_index, q + vect_delta_qi)
            origin_point_neg, insertion_point_neg = update_points_position(model, [0, -1], muscle_index, q - vect_delta_qi)

            # Compute segment lengths for the positive and negative perturbations
            lmt1, _ = compute_segment_length(model, cylinders, q, origin_point_pos, insertion_point_pos, False, False)
            lmt2, _ = compute_segment_length(model, cylinders, q, origin_point_neg, insertion_point_neg, False, False)
        else:
            # If no cylinders are present, calculate length using via points
            mus = model.muscle(muscle_index)
            p_pos = list(update_points_position(model, [n for n in range(len(mus.musclesPointsInGlobal(model, q)))], 
                                                muscle_index, q + vect_delta_qi))
            p_neg = list(update_points_position(model, [n for n in range(len(mus.musclesPointsInGlobal(model, q)))], 
                                                muscle_index, q - vect_delta_qi))

            # Sum up the distances between successive points to calculate muscle-tendon length
            lmt1 = sum(norm(np.array(p_pos[n]) - np.array(p_pos[n+1])) for n in range(len(p_pos)-1))
            lmt2 = sum(norm(np.array(p_neg[n]) - np.array(p_neg[n+1])) for n in range(len(p_neg)-1))

        # Compute the partial derivative using the central difference approximation
        dlmt_dqi = dev_partielle_lmt_qi_points_without_wrapping(lmt1, lmt2, delta_qi)

        # Append the computed derivative to the results list
        dlmt_dq.append(copy.deepcopy(dlmt_dqi))

    print("dlmt_dq =", dlmt_dq)  # Output the results for verification

    return dlmt_dq
 
def plot_all_length_jacobian(model, q_fixed, cylinders, muscle_selected, filename, num_points=100):
    """
    Plot and save a comparison of the variation of dlmt_dq (with wrappings) and dlmt_dq_biorbd (with via points).

    This function generates subplots comparing the derivative of muscle-tendon length (lmt) with respect to joint angles (q)
    computed using two different methods: with wrappings and with via points. It saves the resulting plots to a specified file.

    NOTE: Ensure that len(q_fixed) >= 2 for the function to work correctly.

    Args:
        model (biorbd.Model): The biomechanical model.
        q_fixed (array-like): Array of fixed joint angles chosen by the user.
        cylinders (list): List of Cylinder objects associated with the selected muscle (empty if no cylinders).
        muscle_selected (str): Name of the selected muscle.
        filename (str): Path and filename where the plot will be saved.
        num_points (int, optional): Number of points for plotting (default is 100).

    Returns:
        None: Saves the plot to the specified file.
    """

    print("plot_all_length_jacobian")

    # Check if the plot already exists
    if not os.path.exists(f"{filename}/dlmt_dq.png"):
        # Compute the range of q values for each joint
        q_ranges, q_ranges_names_with_dofs = compute_q_ranges(model)
        
        # Find the index of the selected muscle
        muscle_index = find_index_muscle(model, muscle_selected)
        delta_qi = 1e-10  # Small variation for numerical differentiation
        
        # Determine the layout of subplots
        row_size, col_size = compute_row_col(len(q_ranges), 3)
        fig, axs = plt.subplots(row_size, col_size, figsize=(15, 10))
        
        # Iterate over each joint angle to compute its derivative
        for q_index in range(len(q_ranges)):
            q = copy.deepcopy(q_fixed)
            qs = []
            dlmt_dqis = []
            dlmt_dq_biorbds = []

            # Vary q_index between its min and max range
            for k in range(num_points + 1):
                print("q_index =", q_index, "; k =", k)

                # Compute the current value of qi within its range
                qi = k * ((q_ranges[q_index][1] - q_ranges[q_index][0]) / num_points) + q_ranges[q_index][0]
                q[q_index] = qi

                # Update the model with the new q values and compute the jacobian
                model.updateMuscles(q)
                model.UpdateKinematicsCustom(q)
                dlmt_dq_biorbd = model.musclesLengthJacobian().to_array()

                # Compute the partial derivative of lmt with respect to q
                dlmt_dq = compute_dlmt_dq(model, q, cylinders, muscle_index, delta_qi)

                # Store the results
                qs.append(qi)
                dlmt_dqis.append(dlmt_dq)
                dlmt_dq_biorbds.append(dlmt_dq_biorbd[muscle_index])

            # Determine the subplot position
            row = q_index // 3
            col = q_index % 3

            # Plot dlmt_dq_biorbd and dlmt_dq for each component
            for i in range(len(dlmt_dq)):
                if i == 0:
                    axs[row, col].plot(qs, [dlmt_dqi_biorbd[i] for dlmt_dqi_biorbd in dlmt_dq_biorbds], marker='^',
                                       linestyle='--', color="silver", markersize=2, label=f"dlmt_dq_biorbd")
                else:
                    axs[row, col].plot(qs, [dlmt_dqi_biorbd[i] for dlmt_dqi_biorbd in dlmt_dq_biorbds], marker='^',
                                       linestyle='--', color="silver", markersize=2)

                axs[row, col].plot(qs, [dlmt_dqi[i] for dlmt_dqi in dlmt_dqis], marker='o',
                                   linestyle='-', markersize=2, label=f"dlmt_dq{i}")

            # Label the subplot
            axs[row, col].set_xlabel(f'q{q_index} Variation', fontsize='smaller')
            axs[row, col].set_ylabel(f'dlmt_dq{q_index}', fontsize='smaller')
            axs[row, col].set_title(f'{q_ranges_names_with_dofs[q_index]}', fontsize='smaller')
            axs[row, col].legend()

        # Add a title to the whole figure
        fig.suptitle(f'dlmt_dq\nq_fixed = {q_fixed}', fontweight='bold')
        plt.tight_layout()
        create_and_save_plot(f"{filename}", "dlmt_dq.png")
        plt.show()

def plot_one_length_jacobian(model, q_fixed, cylinders, muscle_selected, filename, num_points=100):
    """
    Plot and save a comparison of the variation of dlmt_dq (with wrappings) and dlmt_dq_biorbd (with via points).

    This function generates subplots comparing the derivative of muscle-tendon length (lmt) with respect to joint angles (q)
    computed using two different methods: with wrappings and with via points. It saves the resulting plots to a specified file.

    NOTE: Ensure that len(q_fixed) >= 2 for the function to work correctly.

    Args:
        model (biorbd.Model): The biomechanical model.
        q_fixed (array-like): Array of fixed joint angles chosen by the user.
        cylinders (list): List of Cylinder objects associated with the selected muscle (empty if no cylinders).
        muscle_selected (str): Name of the selected muscle.
        filename (str): Path and filename where the plot will be saved.
        num_points (int, optional): Number of points for plotting (default is 100).

    Returns:
        None: Saves the plot to the specified file.
    """

    print("plot_one_length_jacobian")

    # Compute the range of q values for each joint
    q_ranges, q_ranges_names_with_dofs = compute_q_ranges(model)

    # Find the index of the selected muscle
    muscle_index = find_index_muscle(model, muscle_selected)
    delta_qi = 1e-10  # Small variation for numerical differentiation

    # Determine the layout of subplots
    row_size, col_size = compute_row_col(len(q_ranges), 3)

    # Iterate over each joint angle index to generate plots
    for q_index in range(len(q_ranges)):
        
        # Check if the plot already exists
        if not os.path.exists(f"{filename}/dlmt_dq{q_index}.png"):
            q = copy.deepcopy(q_fixed)
            qs = []
            dlmt_dqis = []
            dlmt_dq_biorbds = []

            # Vary q_index between its min and max range
            for k in range(num_points + 1):
                print("q_index =", q_index, "; k =", k)

                # Compute the current value of qi within its range
                qi = k * ((q_ranges[q_index][1] - q_ranges[q_index][0]) / num_points) + q_ranges[q_index][0]
                q[q_index] = qi

                # Update the model with the new q values and compute the jacobian
                model.updateMuscles(q)
                model.UpdateKinematicsCustom(q)
                dlmt_dq_biorbd = model.musclesLengthJacobian().to_array()

                # Compute the partial derivative of lmt with respect to q
                dlmt_dq = compute_dlmt_dq(model, q, cylinders, muscle_index, delta_qi)

                # Store the results
                qs.append(qi)
                dlmt_dqis.append(dlmt_dq)
                dlmt_dq_biorbds.append(dlmt_dq_biorbd[muscle_index])
            
            # Create a figure for the current q_index
            fig_qi, axs_qi = plt.subplots(row_size, col_size, figsize=(15, 10))
            
            # Plot the data for each partial derivative component
            for j in range(len(dlmt_dq)):
                row_qi = j // 3
                col_qi = j % 3

                # Compute the average absolute difference between the two methods
                acc = np.mean(abs(np.array([dlmt_dq_biorbd[j] for dlmt_dq_biorbd in dlmt_dq_biorbds]) - 
                                  np.array([dlmt_dq[j] for dlmt_dq in dlmt_dqis])))

                # Plot dlmt_dq_biorbd and dlmt_dq
                axs_qi[row_qi, col_qi].plot(qs, [dlmt_dq_biorbd[j] for dlmt_dq_biorbd in dlmt_dq_biorbds], 
                                           marker='^', linestyle='--', color="silver", 
                                           markersize=2, label=f"dlmt_dq_biorbd{q_index}[{j}]")
                axs_qi[row_qi, col_qi].plot(qs, [dlmt_dq[j] for dlmt_dq in dlmt_dqis], marker='o', linestyle='-', 
                                           markersize=2, label=f"dlmt_dq{q_index}[{j}]")
                
                # Set labels, titles, and legend
                axs_qi[row_qi, col_qi].set_xlabel(f'q{q_index} Variation', fontsize='smaller')
                axs_qi[row_qi, col_qi].set_ylabel(f'dlmt_dq{q_index}[{j}]', fontsize='smaller')
                axs_qi[row_qi, col_qi].set_title(f'{q_ranges_names_with_dofs[q_index]}[{j}] - acc = {acc:.6f}', fontsize='smaller')
                axs_qi[row_qi, col_qi].legend()
            
            # Add a title to the whole figure
            fig_qi.suptitle(f'dlmt_dq{q_index}\nq_fixed = {q_fixed}', fontweight='bold')
            plt.tight_layout()
            create_and_save_plot(f"{filename}", f"dlmt_dq{q_index}.png")
            plt.show()

def plot_length_jacobian(model, q_fixed, cylinders, muscle_selected, directory_name, num_points=100):
    """
    Generate and save plots of the length Jacobian for a given muscle in a model.

    This function creates plots to visualize the variation of the Jacobian of muscle-tendon length with respect to joint angles.
    It generates both individual plots for each joint angle index and a comprehensive plot comparing all indices.

    Args:
        model (biorbd.Model): The biorbd model.
        q_fixed (list or np.ndarray): The fixed joint coordinates.
        cylinders (list): List of cylinder objects used in the model.
        muscle_selected (int): Index of the selected muscle.
        directory_name (str): Directory name where plots will be saved.
        num_points (int, optional): Number of points for the plot. Defaults to 100.
    """
    
    # Create a directory to save the plots if it does not already exist
    directory = f"{directory_name}/plot_length_jacobian"
    create_directory(directory)
    
    # Generate and save the plot for each joint angle index
    plot_one_length_jacobian(model, q_fixed, cylinders, muscle_selected, directory, num_points)
    
    # Generate and save a comprehensive plot for all joint angle indices
    plot_all_length_jacobian(model, q_fixed, cylinders, muscle_selected, directory, num_points)

