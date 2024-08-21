from wrapping.step_1 import switch_frame, transpose_switch_frame
from wrapping.step_2_4 import *
from wrapping.step_3 import determine_if_tangent_points_inactive_single_cylinder
# from wrapping.step_4 import segment_length_single_cylinder
import numpy as np
from wrapping.cadran import determine_if_needed_change_side
from wrapping.plot_cylinder import plot_cadran_double_cylinder, plot_cadran_single_cylinder

# Algorithm
#---------------------------

def single_cylinder_obstacle_set_algorithm(origin_point, insertion_point, Cylinder, plot_cadran = False) :

   """ Calculate the length of a muscle path that wraps around a single cylinder.
    Based on:
    -B.A. Garner and M.G. Pandy, The obstacle-set method for
    representing muscle paths in musculoskeletal models,
    Comput. Methods Biomech. Biomed. Engin. 3 (2000), pp. 1–30.
   -----------------------------------------------------------
   
   This function computes the path length of a muscle segment that wraps around a cylinder.
   It determines the tangent points where the muscle path touches the cylinder and calculates 
   the segment lengths accordingly. 

   Args:
      origin_point (array-like): 3x1 array representing the position of the first point.
      insertion_point (array-like): 3x1 array representing the position of the second point.
      Cylinder (object): An instance of a Cylinder class, which should have 'radius', 'side', and 'matrix' attributes.
      plot_cadran (bool, optional): If True, plots the arrangement of the cylinder and tangent points. Defaults to False.

   Returns:
      tuple: A tuple containing:
         - v1o (array): 3x1 array position of the first obstacle tangent point in the conventional frame.
         - v2o (array): 3x1 array position of the second obstacle tangent point in the conventional frame.
         - obstacle_tangent_point_inactive (bool): Indicates if the tangent points are inactive.
         - segment_length (float): The length of the path segments around the cylinder.
   """

   # ------
   # Step 1: Determine the cylinder radius and adjust based on the side
   # ------
   r = Cylinder.radius * Cylinder.side

   # Express the origin and insertion points in the cylinder's frame of reference
   P_cylinder_frame, S_cylinder_frame = transpose_switch_frame([origin_point, insertion_point], Cylinder.matrix)

   # ------
   # Step 2: Find the tangent points of the cylinder where the muscle path touches
   # ------
   v1, v2 = find_tangent_points(P_cylinder_frame, S_cylinder_frame, r)
   
   # Check if tangent points are valid; if not, set them to the origin
   if np.isnan(np.array(v1)).any() or np.isnan(np.array(v2)).any():
      v1 = np.array([0.0, 0.0, 0.0])
      v2 = np.array([0.0, 0.0, 0.0])

   # ------
   # Step 3: Determine if the tangent points are inactive (i.e., if the muscle does not wrap around the cylinder)
   # ------
   obstacle_tangent_point_inactive = determine_if_tangent_points_inactive_single_cylinder(v1, v2, r)

   # ------
   # Step 4: Compute the length of the path segments considering the tangent points
   # ------
   segment_length = segment_length_single_cylinder(obstacle_tangent_point_inactive, P_cylinder_frame, S_cylinder_frame, 
                                                   v1, v2, r)

   # ------
   # Step 5: Convert the tangent points back to the original frame of reference
   # ------
   v1o, v2o = switch_frame([v1, v2], Cylinder.matrix)
   
   # Optionally plot the cylinder and tangent points
   if plot_cadran:
      plot_cadran_single_cylinder(P_cylinder_frame[:2], S_cylinder_frame[:2], Cylinder, v1[:2], v2[:2], 
                                  obstacle_tangent_point_inactive)

   return v1o, v2o, obstacle_tangent_point_inactive, segment_length

def double_cylinder_obstacle_set_algorithm(P, S, Cylinder_U, Cylinder_V, plot_cadran = False) :

   """ Calculate the length of a muscle path wrapping around two cylinders.
    Based on:
    -B.A. Garner and M.G. Pandy, The obstacle-set method for
    representing muscle paths in musculoskeletal models,
    Comput. Methods Biomech. Biomed. Engin. 3 (2000), pp. 1–30.
   -----------------------------------------------------------

   This function computes the path length of a muscle segment that wraps around two cylinders. 
   It determines tangent points for each cylinder and calculates the segment lengths accordingly. 

   Args:
      P (array-like): 3x1 array representing the position of the first point.
      S (array-like): 3x1 array representing the position of the second point.
      Cylinder_U (object): An instance of a Cylinder class for the first cylinder (U) with 'radius', 'side', and 'matrix' attributes.
      Cylinder_V (object): An instance of a Cylinder class for the second cylinder (V) with 'radius', 'side', and 'matrix' attributes.
      plot_cadran (bool, optional): If True, plots the arrangement of the cylinders and tangent points. Defaults to False.

   Returns:
      tuple: A tuple containing:
         - Qo (array): 3x1 array position of the first obstacle tangent point in the conventional frame.
         - Go (array): 3x1 array position of the second obstacle tangent point in the conventional frame.
         - Ho (array): 3x1 array position of the third obstacle tangent point in the conventional frame.
         - To (array): 3x1 array position of the fourth obstacle tangent point in the conventional frame.
         - Q_G_inactive (bool): Indicates if the tangent points Q and G are inactive.
         - H_T_inactive (bool): Indicates if the tangent points H and T are inactive.
         - segment_length (float): The length of the path segments around the cylinders.
    """

   # ------
   # Step 1: Determine the radius for each cylinder and adjust based on the side
   # ------
   r_U = Cylinder_U.radius * Cylinder_U.side
   r_V = Cylinder_V.radius * Cylinder_V.side

   # Convert points P and S to the frame of reference for each cylinder
   P_U_cylinder_frame, S_U_cylinder_frame = transpose_switch_frame([P, S], Cylinder_U.matrix)
   P_V_cylinder_frame, S_V_cylinder_frame = transpose_switch_frame([P, S], Cylinder_V.matrix)

   # ------
   # Step 2
   # ------
   Q, G, H, T = find_tangent_points_iterative_method(P, S, P_U_cylinder_frame,P_V_cylinder_frame, S_U_cylinder_frame, 
                                                     S_V_cylinder_frame, r_V, r_U,  Cylinder_U.matrix, Cylinder_V.matrix)
   
   H_T_inactive = determine_if_tangent_points_inactive_single_cylinder(H, T, r_V)

   # if H_T_inactive and determine_if_needed_change_side(S_V_cylinder_frame, np.array([0.0246330366, -0.0069265376, -0.0000168612])): 
   if H_T_inactive and determine_if_needed_change_side(S_V_cylinder_frame, np.array([0.0179188682, -0.0181428819, 0.02])) == True: 
      # If there is no wrapping on the second cylinder
      # but there should be, switch sides and try again.
      # BIG WARNING: The point selected for the cadran is arbitrary! It cannot be generalized.
      Cylinder_V.change_side()
      r_V = Cylinder_V.radius * Cylinder_V.side
      Q, G, H, T = find_tangent_points_iterative_method(P, S, P_U_cylinder_frame,P_V_cylinder_frame, S_U_cylinder_frame, S_V_cylinder_frame, r_V, r_U,  Cylinder_U.matrix, Cylinder_V.matrix)
      Cylinder_V.change_side()
      # r_V = Cylinder_V.radius * Cylinder_V.side
      
   # elif H_T_inactive == False and determine_if_needed_change_side_2(S_V_cylinder_frame, np.array([0.0179188682, -0.0181428819, 0.02])) == True: 
   #    print("yop")
   #    Cylinder_V.change_side()
   #    r_V = Cylinder_V.radius * Cylinder_V.side
   #    Q, G, H, T = find_tangent_points_iterative_method(P, S, P_U_cylinder_frame,P_V_cylinder_frame, S_U_cylinder_frame, S_V_cylinder_frame, r_V, r_U,  Cylinder_U.matrix, Cylinder_V.matrix)
   #    Cylinder_V.change_side()
      
   # if H_T_inactive == False and determine_if_needed_change_side(S_V_cylinder_frame, np.array([0.0246330366, -0.0069265376, -0.0000168612])) == True: 
   #    Cylinder_V.change_side()
   #    Q, G, H, T = find_tangent_points_iterative_method(P, S, P_U_cylinder_frame,P_V_cylinder_frame, S_U_cylinder_frame, S_V_cylinder_frame, r_V, r_U,  Cylinder_U.matrix, Cylinder_V.matrix)
   #    Cylinder_V.change_side()
   
   # ------
   # Step 3: Determine if tangent points Q and G are inactive
   # ------
   
   Q_G_inactive = determine_if_tangent_points_inactive_single_cylinder(Q, G, r_U)

   # Recalculate tangent points if needed
   if H_T_inactive==True :
      Q, G = find_tangent_points(P_U_cylinder_frame, S_U_cylinder_frame, r_U)
      Q_G_inactive = determine_if_tangent_points_inactive_single_cylinder(Q, G, r_U)
      # security to avoid bug in step 4, don't worry, this data will be ignored because of tangent point inside the cylinder
      if np.isnan(Q[-1]) or np.isnan(G[-1]) : 
         Q[-1] = 0.0
         G[-1] = 0.0
      # if point_tangent_inside_cylinder(G, Cylinder_U, Cylinder_V, epsilon=0.0) : 
      #    error_wrapping = True
      #    print("error = ", error_wrapping)
         
   elif Q_G_inactive==True :
      H, T = find_tangent_points(P_V_cylinder_frame, S_V_cylinder_frame, r_V)
      H_T_inactive = determine_if_tangent_points_inactive_single_cylinder(H, T, r_V)
      # security to avoid bug in step 4, don't worry, this data will be ignored because of tangent point inside the cylinder
      if np.isnan(Q[-1]) or np.isnan(G[-1]) : 
         Q[-1] = 0.0
         G[-1] = 0.0

    # ------
    # Step 4: Calculate the segment length around both cylinders
    # ------

   segment_length = segment_length_double_cylinder(Q_G_inactive, H_T_inactive, P, S, P_U_cylinder_frame, P_V_cylinder_frame, S_U_cylinder_frame, S_V_cylinder_frame, Q, G, H, T, r_U, r_V, Cylinder_U.matrix, Cylinder_V.matrix)
  
   # ------
   # Step 5: Convert tangent points back to the conventional frame
   # ------
   Qo, Go = switch_frame([Q, G], Cylinder_U.matrix)
   Ho, To = switch_frame([H, T], Cylinder_V.matrix)
   
   if plot_cadran == True : 
      plot_cadran_double_cylinder([P_U_cylinder_frame[:2], P_V_cylinder_frame[:2]], [S_U_cylinder_frame[:2], 
        S_V_cylinder_frame[:2]], [Cylinder_U, Cylinder_V], [Q[:2], H[:2]], [G[:2], T[:2]], 
                                  [Q_G_inactive, H_T_inactive])
   
   return Qo, Go, Ho, To, Q_G_inactive, H_T_inactive, segment_length
