# Functions for Step 3
#----------------------

def determine_if_tangent_points_inactive_single_cylinder(v1,v2, r) :

  """  Determine if tangent points v1 and v2 are inactive
    
    /!\ Differences with the B.A. Garner and M.G. Pandy paper !
      if Det < 0 : orientation is clockwise
      so for a side right-handed (side = 1), we need actived tangent points
      so, if Det*r < 0 ==> determine_if_tangent_points_inactive = False
      (and not "True" as originally presented in the paper)
    
    INPUT
    - v1 : array 3*1 position of the first obstacle tangent point
    - v2 : array 3*1 position of the second obstacle tangent point
    - r : radius of the cylinder * side
    
    OUTPUT
    - obstacle_tangent_point_inactive: bool True if tangent points are inactive --> Muscle path is straight line from 
                                                                                    origin point to final point
                                            False if tangent points are active --> Muscle passes by the two tangent 
                                                                                    points"""

  if (r*(v1[0]*v2[1] - v1[1]*v2[0])<0) :
    return False
  else :
    return True