import numpy as np
import matplotlib as plt
import matplotlib.pyplot as plt

def compute_row_col(sum, div) : 
    """Compute ideal row and col for subplots
    
    Args : 
    - sum : int, sum of all number of plot to do in the figure
    - div : int, number of col max 
    
    Returns : 
    - row : int, number of row for subplot
    - col : int, number of col for subplot """
    
    row = sum//div
    if sum%div != 0 :
        row+=1
    
    return row, min(sum, div)

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

def plot_mvt_discontinuities_in_red(i, qs, segment_lengths, to_remove) : 
    
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
