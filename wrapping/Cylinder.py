import numpy as np
from wrapping.step_1 import find_cylinder_frame, find_matrix

class Cylinder:
    def __init__(self, radius, side, matrix=None, segment=None):
        # segment : 
        # ['thorax', 'spine', 'clavicle_effector_right', 'clavicle_right', 'scapula_effector_right', 'scapula_right', 
        # 'humerus_right', 'ulna_effector_right', 'ulna_right', 'radius_effector_right', 'radius_right', 'hand_right']

        self.radius = radius
        self.side = side
        self.matrix = matrix if matrix is not None else np.identity(4)
        self.segment = segment
        
    def __init__(self, radius, side, c1, c2, segment=None):
        self.radius = radius
        self.side = side
        self.matrix = find_matrix(find_cylinder_frame([c1, c2]), (c1 + c2)/2)
        self.segment = segment

    def __str__(self):
        return (f"Cylinder(radius={self.radius}, "
                f"matrix=\n{self.matrix}, "
                f"segment{self.segment})")