import numpy as np
import matplotlib.pyplot as plt


p0 = np.array([1.0, 1.0])
p1 = np.array([2.0, 2.0])
p2 = np.array([3.0, 3.0])

def find_if_data_ignored(cylinders, points, bool_inactive) : 
   for k in range (len(cylinders)) : 
      if cylinders[k].compute_if_tangent_point_in_cylinder(points[2*k], points[2*k+1], bool_inactive[k]) : 
         return True
   
   return False


def find_discontinute(x, y) : 
    # p_mean = np.mean([p0, p2], axis=0)
    print("")

    # x = x[sorted_indices]
    # y = y[sorted_indices]

    # Calculer les distances entre les points consécutifs
    distances = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    print("distances = ", distances)

    # Définir un seuil pour détecter les discontinuités
    # threshold = 0.08  # Vous pouvez ajuster ce seuil selon le contexte
    threshold = np.mean(distances)
    # Détecter les discontinuités
    discontinuities = np.where(distances > threshold + 0.002)[0]

    if len(discontinuities) != 0 : 
        # Afficher les résultats
        plt.figure(figsize=(10, 6))
        plt.plot(x, y, marker='o')
        for idx in discontinuities:
            plt.plot(x[idx:idx+2], y[idx:idx+2], 'r', linewidth=2)  # Marquer les discontinuités en rouge
        plt.title('Analyse de la continuité de la courbe')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.grid(True)
        plt.show()
        
    return discontinuities


# def data_to_remove(discontinuite, size, range = 10) : 
#     # size = 50
#     # faire discontinuites - 10 et +10
#     min = discontinuite - range
#     if min < 0 : 
#         min = 0
#     max = discontinuite + range 
#     if max > size -1 : 
#         max = size -1
#     return [min, max]

def data_to_remove(discontinuite, size, range = 10):
    min_index = max(0, discontinuite - range)
    max_index = min(size - 1, discontinuite + range)
    return [min_index, max_index]

def find_discontinuites_from_error_wrapping(position, size, range = 5) : 
    # attention, fait comme si il n'a avait quune seul fois un opb de wrapping
    
    min_index = max(0, min(position) - range)
    max_index = min(size - 1, max(position) + range)
    return [min_index, max_index]
    

         