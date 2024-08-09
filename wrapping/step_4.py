# import numpy as np
# from scipy.linalg import norm

# def compute_length_v1_v2_xy(v1,v2, r) :

#   """Compute xy coordinates of segment lengths in plane
  
#   Args
#   - v1 : array 3*1 position of the first obstacle tangent point
#   - v2 : array 3*1 position of the second obstacle tangent point
#   - r : radius of the cylinder * side
  
#   Returns
#   - ||v1v2||(x,y) : xy coordinates of segment lengths in plane"""

#   if r == 0:
#     raise ValueError("Please choose an other radius, positive or negative are accepted. You musn't have r=0")

#   return np.absolute(r*np.arccos(1.0-((v1[0]-v2[0])**2+(v1[1]-v2[1])**2)/(2*r**2)))

# # Functions for Step 4
# #----------------------

# def compute_length_v1_v2(v1,v2, v1_v2_length_xy) :

#   """Compute length of path segments v1 v2
  
#   Args
#   - v1 : array 3*1 position of the first obstacle tangent point
#   - v2 : array 3*1 position of the second obstacle tangent point
#   - v1_v2_length_xy : xy coordinates of segment lengths in plane
  
#   Returns
#   - ||v1v2|| : length of path segments between v1 and v2"""

#   return np.sqrt(v1_v2_length_xy**2+(v2[2]-v1[2])**2)

# def segment_length_single_cylinder(obstacle_tangent_point_inactive, origin_point, final_point, v1, v2, r) :

#   """Compute length of path segments
  
#   Arg
#   - obstacle_tangent_point_inactive : bool determine if v1 and v1 or inactive (True) or not (False)
#   - origin_point : array 3*1 position of the first point
#   - final_point : array 3*1 position of the second point
#   - v1 : array 3*1 position of the first obstacle tangent point
#   - v2 : array 3*1 position of the second obstacle tangent point
#   - r : radius of the cylinder * side
  
#   Returns
#   - segment_length : length of path segments"""

#   if (obstacle_tangent_point_inactive == True) : # Muscle path is straight line from origin_point to final_point
#    segment_length = norm(np.array(final_point)-np.array(origin_point))

#   else :
#    v1_v2_length_xy = compute_length_v1_v2_xy(v1, v2, r)
#    v1_v2_length = compute_length_v1_v2(v1,v2, v1_v2_length_xy)
#    segment_length = norm (v1 - np.array(origin_point)) + v1_v2_length + norm(np.array(final_point) - v2)

#   return segment_length