import numpy as np
import matplotlib.pyplot as plt

def find_if_data_ignored(cylinders, points, bool_inactive) : 
    """Find if the cylinder pass throught origin/insertion point
    Note that with two cylinders and iterative method, an origin/insertion point 
    could be the tangent point of the other cylinder

    INPUTS : 
    - cylinders : list of all cylinder (Cylinder) of muscle
    - points : [origin_point, insertion_point], list of coordinates array 1*3
    - bool_inactive : bool, True if tangent points are inactive
    
    OUTPUT 
    data_ignored : bool
        if any cylinders pass throught origin/insertion point, the data must be ignored 
    """
    for k in range (len(cylinders)) : 
        if cylinders[k].compute_if_tangent_point_in_cylinder(points[2*k], points[2*k+1], bool_inactive[k]) : 
            return True
    
    return False

def find_discontinuities_from_error_wrapping_range(positions, size, range = 5) : 
    """Compute all points to remove relatively to the index "discontinuity
    range --> cut some point before and some point after the discontinuity
    
    WARNING : we suppose here there is only one group of "True positions"
    So, if the cylinder pass throught origin/insertion point more than 1 time, 
    all datas beetween "True positions" groups will be also remove ... 
    
    INPUTS : 
    - positions : [int], indexes of the generated data where a discontinuity was founded (wrapping error)
    - size : int, size of the generated mouvement
    - range : int (default = 5), num of index to remove before and after the discontinuity
    
    OUTPUT : 
    - [min_index, max_index] : [int, int], index min and index max to remove """
    
    min_index = max(0, min(positions) - range)
    max_index = min(size - 1, max(positions) + range)
    return [min_index, max_index]


def find_discontinuty(x, y, epsilon = 0.002, plot_discontinuities = False) : 
    """Find all discontinuities of a curve with distance beetween points
    INPUTS : 
    - x : [float], x axis (qi)
    - y : [float], y axis (muscle length)
    - epsilon : float (default = 0.002), a small value to add at theshold
    - plot_discontinuities : bool (default = False) plot the mouvement with discontinuity
    
    OUTPUT : 
    - discontinuities : [int], list of all index where a discontinuity is founded"""

    # Compute distances beetween consecutives points
    distances = np.sqrt(np.diff(x)**2 + np.diff(y)**2)

    threshold = np.mean(distances)
    discontinuities = np.where(distances > threshold + epsilon)[0]

    if len(discontinuities) != 0 and plot_discontinuities : 
        plt.figure(figsize=(10, 6))
        plt.plot(x, y, marker='o')
        for idx in discontinuities:
            plt.plot(x[idx:idx+2], y[idx:idx+2], 'r', linewidth=2)  # Discontinuities are in red
        plt.title(f'Muscle Length as a Function of q Values')
        plt.xlabel(f'q')
        plt.ylabel('Muscle_length')
        plt.grid(True)
        plt.show()
        
    return discontinuities

def data_to_remove_range(discontinuity, size, range = 5):
    """Compute all points to remove relatively to the index "discontinuity
    range --> cut some point before and some point after the discontinuity"
    INPUTS : 
    - discontinuity : int, index of the generated data where a discontinuity was founded (mvt not physiological)
    - size : int, size of the generated mouvement
    - range : int (default = 5), num of index to remove before and after the discontinuity
    
    OUTPUT : 
    - [min_index, max_index] : [int, int], index min and index max to remove """
    
    min_index = max(0, discontinuity - range)
    max_index = min(size - 1, discontinuity + range)
    return [min_index, max_index]

def data_to_remove_part(discontinuity, qs, size, range = 5):
    """Compute all points to remove relatively to the index "discontinuity
    part --> keep only the more physiological datas"
    INPUTS : 
    - discontinuity : int, index of the generated data where a discontinuity was founded (mvt not physiological)
    - size : int, size of the generated mouvement
    - range : int (default = 5), num of index to remove before and after the discontinuity
    
    OUTPUT : 
    - [min_index, max_index] : [int, int], index min and index max to remove """
    if qs[discontinuity] < 0 : 
        return [0, discontinuity + range]
    else : 
        return [discontinuity - range, size - 1]
    

