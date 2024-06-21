import numpy as np
from wrapping.step_1 import find_cylinder_frame, find_matrix, switch_frame, transpose_switch_frame
from scipy.spatial.transform import Rotation as R

class Cylinder:
    def __init__(self, radius, side, c1, c2, matrix, segment = None, segment_index = None, gcs_seg_0 = None):
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
        self.c1_initial = c1
        self.c2_initial = c2
        self.c1 = c1
        self.c2 = c2
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
        
        return cls(radius, side, c1, c2, matrix, segment)
    
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
        return cls(radius, side, c1, c2, matrix, segment)
        
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
        
        print("self.c1 = ", self.c1)
        self.c1, self.c2 =  origin - d * unit_AB, origin + d * unit_AB
        print("self.c1 = ", self.c1)
    
    def compute_new_matrix_segment(self, model, q) :
        """ Compute the matrix with new q
        - model : model 
        - q : array 4*2, refer to humerus segment"""
        
        matrix_rot_zy = np.array([[0,0,1,0], [1,0,0,0], [0,1,0,0], [0,0,0,1]])
        
        gcs_seg = [gcs.to_array() for gcs in model.allGlobalJCS(q)][self.segment_index]
        self.matrix = np.dot(gcs_seg, np.dot(np.linalg.inv(self.gcs_seg_0), self.matrix_initial))
        
        print("Insertion doit être là = ", switch_frame([0.016, -0.0354957, 0.005], gcs_seg)) # oui
        
        # truc bizarre
        # # self.matrix = np.dot(np.linalg.inv(self.gcs_seg_0), self.matrix_initial)
        # self.c1 = switch_frame([0, 0.05, 0], gcs_seg)
        # self.c2 = switch_frame([0,-0.05, 0], gcs_seg)
        
        # # self.c1 = np.dot(matrix_rot_zy[0:3, 0:3], self.c1)
        # # self.c2 = np.dot(matrix_rot_zy[0:3, 0:3], self.c2)
        # print("")
        # frame = find_cylinder_frame([self.c1, self.c2])
        # midpoint = (self.c1 + self.c2) / 2
        # self.matrix = find_matrix(frame, midpoint)
        
    def compute_new_matrix_segment2(self, model, q) :
        """ Compute the matrix with new q
        - model : model 
        - q : array 4*2, refer to humerus segment"""
        
        matrix_rot_zy = np.array([[0,0,1,0], [1,0,0,0], [0,1,0,0], [0,0,0,1]])
        
        gcs_seg = [gcs.to_array() for gcs in model.allGlobalJCS(q)][self.segment_index]
        self.matrix = np.dot(gcs_seg, np.dot(np.linalg.inv(self.gcs_seg_0), self.matrix_initial))
        
        print("Insertion doit être là = ", switch_frame([0.016, -0.0354957, 0.005], gcs_seg)) # oui
        
        # truc bizarre
        # self.matrix = np.dot(np.linalg.inv(self.gcs_seg_0), self.matrix_initial)
        # self.c1 = switch_frame([0, -0.05, 0], gcs_seg)
        # self.c2 = switch_frame([0, 0.05, 0], gcs_seg)
        
        self.c1 = transpose_switch_frame(self.c1_initial, self.gcs_seg_0)
        self.c2 = transpose_switch_frame(self.c2_initial, self.gcs_seg_0)
        # ces points, c1 et c2 sont maintenant dans le repere local de humerus (verifie)
        
        # print("les points du cylindres en repere local humerus sont : ")
        # print("self.c1 = ", self.c1)
        # print("self.c2 = ", self.c2)
        
        self.c1 = switch_frame(self.c1, gcs_seg)
        self.c2 = switch_frame(self.c2, gcs_seg)
        # pour mettre dans le global avec nouveau gcs seg (ok)
        # print("les points du cylindres en repere global humerus sont : ")
        # print("self.c1 = ", self.c1)
        # print("self.c2 = ", self.c2)
        
        # self.c1 = np.dot(matrix_rot_zy[0:3, 0:3], self.c1)
        # self.c2 = np.dot(matrix_rot_zy[0:3, 0:3], self.c2)
        print("")
        frame = find_cylinder_frame([self.c1, self.c2])
        midpoint = (self.c1 + self.c2) / 2
        self.matrix = find_matrix(frame, midpoint)
        
        # self.change_side2(self.rotation_direction(self.matrix_initial, self.matrix))
        # self.change_side2(self.rotation_direction(self.gcs_seg_0, gcs_seg))
        
        print("radius = ", self.radius)
    
        
    def compute_matrix_rotation_zy(self, matrix_rot_zy) : 
        """ Compute transformation of matrix (rotation z --> y)
        - matrix_rot_zy : rotation matrix z --> y"""
        
        self.matrix = np.dot(matrix_rot_zy, self.matrix)
        
        # self.c1 = transpose_switch_frame(self.c1, matrix_rot_zy)
        # self.c2 = transpose_switch_frame(self.c2, matrix_rot_zy)
        
        # self.c1_initial = switch_frame(self.c1_initial, np.transpose(matrix_rot_zy))
        # self.c2_initial = switch_frame(self.c2_initial, np.transpose(matrix_rot_zy))
        
    def compute_seg_index_and_gcs_seg_0(self, q_initial, model, segment_names) :
        """Compute gcs 0 (inital) and index of cylinder(s)
        
        - model : model
        - q_initial : array 4*2, refer to humerus segment (initial)
        - segment_name : list of all segments name of the model"""
        
        self.segment_index = segment_names.index(self.segment) 
        self.gcs_seg_0 = [gcs.to_array() for gcs in model.allGlobalJCS(q_initial)][self.segment_index] 
        
    # def correcte_side(self, q) :
        
    #     # if q0 == 0 : 
    #     #     self.side = 1
    #     # else : 
    #     #     self.side = q0 / abs(q0)
    #     if q[2] < 0.355 and q[0] < -0.367 and q[1] > -1.497: 
    #         self.side = -1
    #     else : 
    #         self.side = 1
    #     # if q[1] > -1.49725651 and q[0] != 0 :#and q[2] != 2 :  # -1.49725651
    #     #     self.side = q[0] / abs(q[0]) #* q[2] / abs(q[2])
    #     # # elif q[1] < -1.49725651 and q[2] != 0 : # -1.49725651
    #     # #     self.side = q[2] / abs(q[2])
    #     # else : 
        #     self.side = 1

    def change_raidus(self, new_radius) : 
        self.raidus = new_radius
        
    def change_side(self) : 
        # self.side = new_side
        self.side = self.side * -1
        
    def change_side2(self, new_side) : 
        self.side = new_side
        # self.side = self.side * -1
        

        
    def extract_angles(self, matrix):
        """Extract rotation angles around x and y from a 4x4 rotation-translation matrix."""
        # We assume the matrix is in the form of:
        # [ R | T ]
        # [ 0 | 1 ]
        R = matrix[:3, :3]

        # Extract angles using atan2 for better numerical stability
        theta_x = np.arctan2(R[2, 1], R[2, 2])
        theta_y = np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2))
        theta_z = np.arctan2(R[1, 0], R[0, 0])

        return theta_x, theta_y, theta_z

    def rotation_direction(self, matrix_initial, matrix_update):
        """Determine the rotation direction around the z-axis between two matrices.

        INPUT:
        - matrix_initial: 4x4 initial rotation-translation matrix
        - matrix_update: 4x4 updated rotation-translation matrix

        OUTPUT:
        - 1 if the rotation is clockwise
        - -1 if the rotation is counterclockwise
        """
        # Extract the rotation angles from both matrices
        _, _, theta_z_initial = self.extract_angles(matrix_initial)
        _, _, theta_z_update = self.extract_angles(matrix_update)

        # Calculate the change in the z rotation angle
        delta_theta_z = theta_z_update - theta_z_initial

        # Normalize the delta_theta_z to be within -pi to pi
        delta_theta_z = (delta_theta_z + np.pi) % (2 * np.pi) - np.pi

        # Determine the direction
        if delta_theta_z > 0:
            direction = 1
        else:
            direction = -1

        # If the absolute change is greater than pi, reverse the direction
        if abs(delta_theta_z) > np.pi:
            direction *= -1

        return direction

    def __str__(self):
        return (f"Cylinder(radius = {self.radius}, "
                f"side=\n{self.side}, "
                f"matrix_initial=\n{self.matrix_initial}, "
                f"matrix=\n{self.matrix}, "
                f"segment : {self.segment},"
                f"segment_index=\n{self.segment_index}, "
                f"gcs_seg_0=\n{self.gcs_seg_0})")
        