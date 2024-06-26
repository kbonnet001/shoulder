import math
import numpy as np

def angle_between_points(p1, p2):
    # Calcul du produit scalaire
    dot_product = p1[0] * p2[0] + p1[1] * p2[1]
    
    # Calcul des normes des vecteurs
    norm_u = math.sqrt(p1[0]**2 + p1[1]**2)
    norm_v = math.sqrt(p2[0]**2 + p2[1]**2)
    
    # Calcul du cosinus de l'angle
    cos_theta = dot_product / (norm_u * norm_v)
    
    # Correction des imprécisions numériques
    cos_theta = max(min(cos_theta, 1.0), -1.0)
    
    # Calcul de l'angle en radians
    angle_rad = math.acos(cos_theta)
    
    # Conversion de l'angle en degrés
    angle_deg = math.degrees(angle_rad)
    
    return angle_deg

def roto_trans_matrix(angle):
    # Rotation matrix to rotate around the origin
    R = np.array([
        [np.cos(angle), -np.sin(angle), 0.0, 0.0],
        [np.sin(angle),  np.cos(angle), 0.0, 0.0],
        [0.0,             0.0, 1.0 , 0.0], 
        [0.0,             0.0, 0.0 , 1.0]
    ])
    
    return R

def apply_transformation(p, M):
    # Create the homogeneous coordinate vector for the point (x, y, z)
    point = np.array([p[0], p[1], p[2], 1.0])
    
    # Apply the transformation matrix
    transformed_point = np.dot(M, point)
    
    # Return the transformed point
    return transformed_point[:3]  # Only the x, y, and z components

def determine_if_needed_change_side(point, point_ref_q_initial) :
    angle = angle_between_points(point_ref_q_initial[:2], [1.0, 0.0] ) 
    angle_rad = math.radians(angle)
    R = roto_trans_matrix(angle_rad)
    # R = np.array([[ 0.96266626 ,-0.27069111  ,0.      ,    0.        ],
    #               [ 0.27069111 , 0.96266626 , 0.      ,    0.        ],
    #               [ 0.         , 0.         , 1.      ,    0.        ],
    #               [ 0.         , 0.         , 0.       ,   1.        ]])
    point = apply_transformation(point, R)
    
    print("point transforme = ", point)

    # if point[1] < 0 and point: 
    #   return True
    # else :
    #   return False
    if point[0] > 0 and point[1] > 0 :
        return True
    else : 
        return False

def determine_if_needed_change_side_2(point, point_ref_q_initial) :
    angle = angle_between_points(point_ref_q_initial[:2], [1.0, 0.0] ) 
    angle_rad = math.radians(angle)
    R = roto_trans_matrix(angle_rad)
    # R = np.array([[ 0.96266626 ,-0.27069111  ,0.      ,    0.        ],
    #               [ 0.27069111 , 0.96266626 , 0.      ,    0.        ],
    #               [ 0.         , 0.         , 1.      ,    0.        ],
    #               [ 0.         , 0.         , 0.       ,   1.        ]])
    point = apply_transformation(point, R)
    
    print("point transforme = ", point)

    # if point[1] < 0 and point: 
    #   return True
    # else :
    #   return False
    if point[0] > 0  :
        return True
    else : 
        return False