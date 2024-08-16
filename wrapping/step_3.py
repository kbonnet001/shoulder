# Functions for Step 3
#----------------------

def determine_if_tangent_points_inactive_single_cylinder(v1,v2, r) :

  """ Determines if tangent points v1 and v2 are inactive.

  Note:
  This implementation differs from the method described in the B.A. Garner and M.G. Pandy paper.
  - If Det < 0, the orientation is clockwise.
  - For a right-handed side (side = 1), we need active tangent points.
  - Therefore, if Det * r < 0, the tangent points are considered active 
    (i.e., determine_if_tangent_points_inactive = False), contrary to what is presented in the original paper.

  Args:
  - v1: 3x1 array, position of the first obstacle tangent point.
  - v2: 3x1 array, position of the second obstacle tangent point.
  - r: radius of the cylinder multiplied by the side (side is a directional factor).

  Returns:
  - obstacle_tangent_point_inactive: A boolean value indicating the status of the tangent points.
    - True: Tangent points are inactive, meaning the muscle path is a straight line from the origin point to the final point.
    - False: Tangent points are active, meaning the muscle passes through the two tangent points.
  """

  if (r*(v1[0]*v2[1] - v1[1]*v2[0])<0) :
    return False
  else :
    return True
