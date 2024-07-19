import numpy as np

def compute_row_col(sum, div) : 
    """Compute ideal row and col for subplots
    
    INPUTS : 
    - sum : int, sum of all number of plot to do in the figure
    - div : int, number of col max 
    
    OUTPUTS : 
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