import numpy as np
from wrapping.step_1 import find_cylinder_frame, find_matrix

class Cylinder:
    def __init__(self, radius, side, matrix=None, segment = None):
        
        # To create a cylinder with a matrix
        #
        # - radius : radius of the cylinder
        # - side (-1 or 1) : Choose 1 for the right-handed side, -1 for the left-handed side (with respect to the z-axis)
        # - matrix : array 4*4 rotation_matrix and vect
        # - segment : segment of the cylinder. Please choose an autorized name in this list: 
        #            ['thorax', 'spine', 'clavicle_effector_right', 'clavicle_right', 'scapula_effector_right', 
        #             'scapula_right', 'humerus_right', 'ulna_effector_right', 'ulna_right', 'radius_effector_right', 
        #             'radius_right', 'hand_right']

        self.radius = radius
        self.side = side
        self.matrix = matrix if matrix is not None else np.identity(4)
        self.segment = segment
        
    def __init__(self, radius, side, c1, c2, segment = None):
        
        # To create a cylinder with two points and create a matrix for this cylinder
        #
        # - radius : radius of the cylinder
        # - side (-1 or 1) : Choose 1 for the right-handed side, -1 for the left-handed side (with respect to the z-axis)
        # - c1 : array 3*1 coordinates of the first circle of the cylinder
        # - c2 : array 3*1 coordinates of the second circle of the cylinder
        # - matrix : array 4*4 rotation_matrix and vect
        # - segment : segment of the cylinder. Please choose an autorized name in this list: 
        #            ['thorax', 'spine', 'clavicle_effector_right', 'clavicle_right', 'scapula_effector_right', 
        #             'scapula_right', 'humerus_right', 'ulna_effector_right', 'ulna_right', 'radius_effector_right', 
        #             'radius_right', 'hand_right']
        
        self.radius = radius
        self.side = side
        self.matrix = find_matrix(find_cylinder_frame([c1, c2]), (c1 + c2)/2)
        self.segment = segment

    def __str__(self):
        return (f"Cylinder(radius = {self.radius}, "
                f"matrix=\n{self.matrix}, "
                f"segment : {self.segment})")