import math
import numpy as np

def angle_between_points(p1, p2):
    """
    Calculate the angle in degrees between two 2D points/vectors.
    
    Args:
        p1 (tuple or list): Coordinates of the first point/vector (x1, y1).
        p2 (tuple or list): Coordinates of the second point/vector (x2, y2).
    
    Returns:
        angle_deg (float): The angle between the two points in degrees.
    """
    
    # Calculate the dot product of the two vectors
    dot_product = p1[0] * p2[0] + p1[1] * p2[1]
    
    # Calculate the norms (magnitudes) of the vectors
    norm_u = math.sqrt(p1[0]**2 + p1[1]**2)
    norm_v = math.sqrt(p2[0]**2 + p2[1]**2)
    
    # Calculate the cosine of the angle using the dot product formula
    cos_theta = dot_product / (norm_u * norm_v)
    
    # Correct numerical imprecision to ensure the value is within the valid range for acos
    cos_theta = max(min(cos_theta, 1.0), -1.0)
    
    # Calculate the angle in radians using the arccosine function
    angle_rad = math.acos(cos_theta)
    
    # Convert the angle from radians to degrees
    angle_deg = math.degrees(angle_rad)
    
    return angle_deg

def roto_trans_matrix(angle):
    """
    Create a 4x4 rotation-translation matrix for rotating around the origin in 2D.
    
    Args:
        angle (float): The angle in radians by which to rotate around the origin.
    
    Returns:
        np.ndarray: A 4x4 rotation matrix.
    """

    # Define the rotation matrix in homogeneous coordinates
    R = np.array([
        [np.cos(angle), -np.sin(angle), 0.0, 0.0],  # Rotation around the Z-axis
        [np.sin(angle),  np.cos(angle), 0.0, 0.0],  # Rotation around the Z-axis
        [0.0,             0.0,          1.0, 0.0],  # No change in the Z direction
        [0.0,             0.0,          0.0, 1.0]   # Homogeneous coordinate
    ])
    
    return R

def apply_transformation(p, M):
    """
    Apply a 4x4 transformation matrix to a 3D point.
    
    Args:
        p (tuple or list): The 3D coordinates of the point (x, y, z).
        M (np.ndarray): The 4x4 transformation matrix.
    
    Returns:
        np.ndarray: The transformed 3D coordinates (x', y', z').
    """
    
    # Create the homogeneous coordinate vector for the point (x, y, z, 1)
    point = np.array([p[0], p[1], p[2], 1.0])
    
    # Apply the transformation matrix to the point
    transformed_point = np.dot(M, point)
    
    # Return only the x, y, and z components of the transformed point
    return transformed_point[:3]

def determine_if_needed_change_side(point, point_ref_q_initial):
    """
    Determine if the point needs to switch sides based on its position after rotation.
    
    Args:
        point (tuple or list): The 3D coordinates of the point (x, y, z).
        point_ref_q_initial (tuple or list): The initial reference point for comparison.
    
    Returns:
        bool: True if the point is in the first quadrant after transformation, False otherwise.
    """

    # Calculate the angle between the reference point and the x-axis
    angle = angle_between_points(point_ref_q_initial[:2], [1.0, 0.0])
    
    # Convert the angle from degrees to radians
    angle_rad = math.radians(angle)
    
    # Generate the rotation matrix based on the calculated angle
    R = roto_trans_matrix(angle_rad)

    # Apply the transformation to the point
    point_transformed = apply_transformation(point, R)
    
    # Debug print to check the transformed point
    print("Transformed point =", point_transformed)

    # Determine if the point is in the first quadrant (both x and y are positive)
    if point_transformed[0] > 0 and point_transformed[1] > 0:
        return True
    else:
        return False

# def determine_if_needed_change_side_2(point, point_ref_q_initial) :
#     angle = angle_between_points(point_ref_q_initial[:2], [1.0, 0.0] ) 
#     angle_rad = math.radians(angle)
#     R = roto_trans_matrix(angle_rad)
#     # R = np.array([[ 0.96266626 ,-0.27069111  ,0.      ,    0.        ],
#     #               [ 0.27069111 , 0.96266626 , 0.      ,    0.        ],
#     #               [ 0.         , 0.         , 1.      ,    0.        ],
#     #               [ 0.         , 0.         , 0.       ,   1.        ]])
#     point = apply_transformation(point, R)
    
#     print("point transforme = ", point)

#     # if point[1] < 0 and point: 
#     #   return True
#     # else :
#     #   return False
#     if point[0] > 0  :
#         return True
#     else : 
#         return False


# def point_to_initial(points, point_ref) :
#     # pas tres tres english
#     # to do a rotation, we want point ref at (1, 0, z)
#     # ans we also want other point (points) moove in the same way
#     #points [] list of point dont point ref
#     angle = angle_between_points(point_ref[:2], [1.0, 0.0] ) 
#     angle_rad = math.radians(angle)
#     R = roto_trans_matrix(angle_rad)

#     transformed_points = []
    
#     for point in points : 
#         point_rot = apply_transformation(point, R)
#         transformed_points.append(point_rot)
#         print("point transforme = ", point_rot)
         
#     return transformed_points
    
   
# def arctan2_corrected(v1, v2) : 
#     v1_rot, v2_rot = point_to_initial([v1, v2], v1)
#     arctan2 = np.arctan2(v2_rot[1], v2_rot[0])
#     print("arctan2 = ", arctan2)
#     if arctan2 < 0 : 
#         # arctan2 = arctan2 + 360 * np.pi/180
#         v1_rot, v2_rot = point_to_initial([v1, v2], v2)
#         arctan2 = np.arctan2(v1_rot[1], v1_rot[0])
#         print("arctan2 + 180 = ", arctan2)
#     return arctan2

# def arctan2_corrected(v1, v2) : 
#     v1_rot, v2_rot = point_to_initial([v1, v2], v1)
#     arctan2 = np.arctan2(v2_rot[1], v2_rot[0])
#     print("arctan2 = ", arctan2)
#     if arctan2 < 0 : 
#         arctan2 = arctan2 + 360 * np.pi/180
#         # v1_rot, v2_rot = point_to_initial([v1, v2], v2)
#         # arctan2 = np.arctan2(v1_rot[1], v1_rot[0])
#         print("arctan2 + 180 = ", arctan2)
#     return arctan2