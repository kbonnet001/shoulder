import numpy as np
from wrapping.step_1 import find_cylinder_frame, find_matrix

class Cylinder:
    def __init__(self, radius, side, matrix, segment = None, segment_index = None, gcs_seg_0 = None):
        """Initialize a cylinder with a transformation matrix
        
        - radius : radius of the cylinder
        - side (-1 or 1) : Choose 1 for the right-handed side, -1 for the left-handed side (with respect to the z-axis)
        - matrix_initial : array 4*4 rotation matrix and translation vector initialy
        - matrix : array 4*4 rotation matrix and translation vector
        - segment : (default = None) segment of the cylinder. Please choose an authorized name in this list: 
                    ['thorax', 'spine', 'clavicle_effector_right', 'clavicle_right', 'scapula_effector_right', 
                     'scapula_right', 'humerus_right', 'ulna_effector_right', 'ulna_right', 'radius_effector_right', 
                     'radius_right', 'hand_right']
        - segment_index : (default = None) int, index position of the segment in the segment list of model
        - gcs_seg_0 : (default = None) array 4*4, rotation matrix and translation vector of the segment initialy"""
        
        self.radius = radius
        self.side = side
        self.matrix_initial = matrix
        self.matrix = matrix
        self.segment = segment
        self.segment_index = segment_index
        self.gcs_seg_0 = gcs_seg_0

    @classmethod
    def from_matrix(cls, radius, side, matrix, segment=None, d = 0.1):
        """
        Create a cylinder with a given transformation matrix.
        
        - radius : radius of the cylinder
        - side (int): Choose 1 for the right-handed side, -1 for the left-handed side (with respect to the z-axis)
        - matrix : array 4*4 rotation matrix and translation vector
        - segment : segment of the cylinder (optional)
        """
        origin = matrix[0:-1,-1]
        z = matrix[2,0:-1]
        
        # Calculer la norme du vecteur AB
        norm_AB = np.linalg.norm(z - origin)
        
        # Calculer le vecteur unitaire dans la direction de AB
        unit_AB = (z - origin) / norm_AB
        
        c1 = origin - d * unit_AB
        c2 = origin + d * unit_AB
        
        return cls(radius, side, matrix, segment)
    
    @classmethod
    def from_points(cls, radius, side, c1, c2, segment=None):
        """Create a cylinder with two points and create a matrix for this cylinder
        
        - radius : radius of the cylinder
        - side (-1 or 1) : Choose 1 for the right-handed side, -1 for the left-handed side (with respect to the z-axis)
        - c1 : array 3*1 coordinates of the first circle of the cylinder
        - c2 : array 3*1 coordinates of the second circle of the cylinder
        - segment : segment of the cylinder. Please choose an authorized name in this list: 
                    ['thorax', 'spine', 'clavicle_effector_right', 'clavicle_right', 'scapula_effector_right', 
                     'scapula_right', 'humerus_right', 'ulna_effector_right', 'ulna_right', 'radius_effector_right', 
                     'radius_right', 'hand_right']"""
        frame = find_cylinder_frame([c1, c2])
        midpoint = (c1 + c2) / 2
        matrix = find_matrix(frame, midpoint)
        return cls(radius, side, matrix, segment)
        
    def rotate_around_axis(self, alpha) :
        """
        Rotate the cylinder around its axis by a given angle.
    
         - alpha : angle in degrees
        """
         
        alpha = np.radians(alpha) # Angle in radians
        rotation_axe_cylinder = np.array([[np.cos(alpha), - np.sin(alpha), 0],
                                        [ np.sin(alpha), np.cos(alpha), 0],
                                        [ 0.       ,  0.        ,  1.    ]])
        new_rotation_matrix = np.dot( self.matrix[:3, :3],(rotation_axe_cylinder))
        
        self.matrix = np.array([[ new_rotation_matrix[0,0], new_rotation_matrix[0,1],new_rotation_matrix[0,2], self.matrix[0,3]],
                                [ new_rotation_matrix[1,0], new_rotation_matrix[1,1],new_rotation_matrix[1,2], self.matrix[1,3]],
                                [ new_rotation_matrix[2,0], new_rotation_matrix[2,1],new_rotation_matrix[2,2], self.matrix[2,3]],
                                 [ 0.        ,  0.        ,  0.        ,  1.        ]])
        return None
        
    def find_points(self, d = 0.01):
        
        origin = self.matrix[0:-1,-1]
        z = self.matrix[2,0:-1]
        
        # Calculer la norme du vecteur AB
        norm_AB = np.linalg.norm(z - origin)
        
        # Calculer le vecteur unitaire dans la direction de AB
        unit_AB = (z - origin) / norm_AB
        
        return origin - d * unit_AB, origin + d * unit_AB
    
    def compute_new_matrix_segment(self, model, q) :
        """ Compute the matrix with new q
        - model : model 
        - q : array 4*2, refer to humerus segment"""
        
        gcs_seg = [gcs.to_array() for gcs in model.allGlobalJCS(q)][self.segment_index]
        self.matrix = np.dot(gcs_seg, np.dot(np.linalg.inv(self.gcs_seg_0), self.matrix_initial))
        
    def compute_matrix_rotation_zy(self, matrix_rot_zy) : 
        """ Compute transformation of matrix (rotation z --> y)
        - matrix_rot_zy : rotation matrix z --> y"""
        
        self.matrix = np.dot(matrix_rot_zy, self.matrix)
        
    def compute_seg_index_and_gcs_seg_0(self, q_inital, model, segment_names) :
        """Compute gcs 0 (inital) and index of cylinder(s)
        
        - model : model
        - q_inital : array 4*2, refer to humerus segment (inital)
        - segment_name : list of all segments name of the model"""
        
        self.segment_index = segment_names.index(self.segment) 
        self.gcs_seg_0 = [gcs.to_array() for gcs in model.allGlobalJCS(q_inital)][self.segment_index] 


    def __str__(self):
        return (f"Cylinder(radius = {self.radius}, "
                f"side=\n{self.side}, "
                f"matrix_initial=\n{self.matrix_initial}, "
                f"matrix=\n{self.matrix}, "
                f"segment : {self.segment},"
                f"segment_index=\n{self.segment_index}, "
                f"gcs_seg_0=\n{self.gcs_seg_0})")
        