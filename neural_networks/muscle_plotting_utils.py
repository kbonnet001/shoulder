import numpy as np
import matplotlib as plt
import matplotlib.pyplot as plt
import math as mat

def compute_row_col(sum) : 
    """Compute ideal row and col for subplots
    
    Args : 
    - sum : int, sum of all number of plot to do in the figure
    
    Returns : 
    - row : int, number of row for subplot
    - col : int, number of col for subplot """
    
    #  div : int, number of col max 
    div = compute_div(sum)
    
    row = sum//div
    if sum%div != 0 :
        row+=1
    
    return row, min(sum, div)

def compute_div(val):
    """
    Compute an adjusted square root value based on the given number.

    Parameters:
    val (float): The number for which the square root will be calculated.

    Returns:
    float: The adjusted square root value.
    """
    # Compute the integer part of the square root of the given value
    int_sqrt = int(mat.sqrt(val))
    
    # Check if the actual square root is between int_sqrt and int_sqrt + 0.5
    if int_sqrt <= mat.sqrt(val) < int_sqrt + 0.5:
        return int_sqrt
    else:
        # If the actual square root is equal to or greater than int_sqrt + 0.5, return int_sqrt + 0.5
        return int_sqrt + 1

def plot_mvt_discontinuities_in_red(i, qs, segment_lengths, to_remove) : 
    """ Plots the muscle lengths as a function of joint angles (qs), 
    highlighting discontinuities in red.

    Args:
    - i (int): Index or identifier for the plot, used in labels.
    - qs (list): List of joint angles.
    - segment_lengths (list): Corresponding muscle lengths for each joint angle.
    - to_remove (list): Indices of discontinuities in the muscle lengths.

    Returns:
    - None: Displays the plot.
    """
    plt.plot(qs, segment_lengths, linestyle='-', color='b')
    qs_plot = [qs[idx] for idx in range(len(qs)) if idx not in to_remove]
    segment_lengths_plot = [segment_lengths[idx] for idx in range(len(segment_lengths)) if idx not in to_remove]
    plt.plot(qs_plot, segment_lengths_plot, marker='o', color='b')
    for idx in to_remove:
        plt.plot(qs[idx:idx+1], segment_lengths[idx:idx+1], marker='x', color='r')  # Discontinuities are in red
    plt.xlabel(f'q{i}')
    plt.ylabel('Muscle_length')
    plt.title(f'Muscle Length as a Function of q{i} Values')
    plt.xticks(qs[::5])
    plt.yticks(segment_lengths[::5]) 
    plt.grid(True)
    plt.show()

# Functions to find num bin for plot hist but do not work well ...
def compute_num_bins(data, x_max, x_min) : 
    # Calculer l'IQR et le nombre de données
    IQR = np.percentile(data, 75) - np.percentile(data, 25)
    N = len(data)

    # Calculer la largeur des bins selon la règle de Freedman-Diaconis
    bin_width = 2 * IQR / np.cbrt(N)

    # Calculer le nombre de bins
    num_bins = int((x_max - x_min) / bin_width)
    
    return num_bins

def sturges_rule(data):
    return int(np.ceil(np.log2(len(data)) + 1))

def rice_rule(data):
    return int(np.ceil(2 * (len(data) ** (1/3))))

def scott_rule(data):
    bin_width = 3.5 * np.std(data) / (len(data) ** (1/3))
    return int((np.max(data) - np.min(data)) / bin_width)

def get_markers(num_groups):
    """
    Generate a list of markers with a length matching the number of groups.

    Args:
        num_groups (int): Number of distinct groups or datasets.

    Returns:
        list: List of marker styles.
    """
    # Define a list of possible marker styles available in matplotlib
    available_markers = ["o", "s", "D", "v", "^", "<", ">", "P", "X", "h", "*", "+", "x", "1", "2", "3", "4", "8", 
                         "|", "_"]
    # If the number of groups is greater than the available markers, cycle through them
    markers = [available_markers[i % len(available_markers)] for i in range(num_groups)]
    return markers