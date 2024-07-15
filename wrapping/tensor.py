from scipy.linalg import norm
from neural_networks.functions_data_generation import update_points_position
import copy
import casadi as ca

def dev_partielle_lmt_qi_points_without_wrapping (lmt1, lmt2, delta_qi) : 
    #p1 p2 point 3D
    # q q len qranges
    # qi index i
    # delta qi tre petite variation de q, attention vect aue des 0 et juste le qi aue l'on vut epsilon
    
    return (lmt1 - lmt2) / (2*delta_qi)

def essai_dlmt_dq(model, q, muscle_index, delta_qi = 1e-4) :
    
    dlmt_dq = []
    
    for i in range(len(q)) : 
        
        # Create vect delta_qi
        q_delta_qi = [0 for k in range(len(q))]
        q_delta_qi[i] = delta_qi
        
        origin_point_pos, insertion_point_pos = update_points_position(model, [0, -1], muscle_index, q + q_delta_qi)
        origin_point_neg, insertion_point_neg = update_points_position(model, [0, -1], muscle_index, q - q_delta_qi)
        
        lmt1 = norm(origin_point_pos - insertion_point_pos)
        lmt2 = norm(origin_point_neg - insertion_point_neg)
        
        dlmt_dqi = dev_partielle_lmt_qi_points_without_wrapping(lmt1, lmt2, delta_qi)
        
        dlmt_dq.append(copy.deepcopy(dlmt_dqi))
    
    return dlmt_dq


# ------------CASADI-------------------

# # Déclaration des variables de décision
# p1 = ca.SX.sym('p1', 3)
# p2 = ca.SX.sym('p2', 3)
# r = ca.SX.sym('r')  # /!\ radius * side cylindre

# # Calcul de la première partie de la fonction
# distance_xy = (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2
# cos_theta = 1.0 - distance_xy / (2 * r**2)
# theta = ca.acos(cos_theta)
# f_part1 = r * theta

# # Calcul de la deuxième partie de la fonction
# f_part2 = (p2[2] - p1[2])**2

# # Somme des parties et racine carrée
# f = ca.sqrt(f_part1 + f_part2)

# print(f)
        
    
    