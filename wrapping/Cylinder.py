import numpy as np
from scipy.linalg import norm
from scipy.spatial.transform import Rotation as R
from wrapping.step_1 import switch_frame, transpose_switch_frame

class Cylinder:
    def __init__(self, radius, side, c1, c2, matrix, point_on_cylinder = False, segment = None, muscle = None, 
                 segment_index = None, gcs_seg_0 = None):
        
        """ Initializes a cylinder with a transformation matrix.

        Args:
        - radius: float, radius of the cylinder.
        - side: int (-1 or 1) indicating the handedness of the side:
            - 1 for the right-handed side
            - -1 for the left-handed side (with respect to the z-axis)
        - c1_initial: 3x1 array, coordinates of the center of the first circle (bottom or top) in the initial 
        configuration (q_initial = 0).
        - c2_initial: 3x1 array, coordinates of the center of the second circle (top or bottom) in the initial 
        configuration (q_initial = 0).
        - matrix_initial: 4x4 array, initial rotation matrix and translation vector.
        - matrix: 4x4 array, current rotation matrix and translation vector.
        - point_on_cylinder: bool (default is False). Set to True if the origin or insertion point is on the surface of 
        the cylinder. 
        Warning: This option may require adjustments for proper functionality. 
        It was originally introduced to prevent the point from entering the cylinder and to adjust the cylinder's 
        radius so that the point remains on the surface. This is particularly useful for PECM2 and PECM3 with 
        two wrapped cylinders. Ensure the correct order of cylinders in the "cylinders" list:
        - Example:
            - cylinders = [cylinder_1, cylinder_2]
            - cylinder_1.point_on_cylinder = True  # Origin point is on the surface
            - cylinder_2.point_on_cylinder = True  # Insertion point is on the surface
        - segment: (default = None) The segment of the cylinder. Choose an authorized name from the following list:
            ['thorax', 'spine', 'clavicle_effector_right', 'clavicle_right', 'scapula_effector_right',
            'scapula_right', 'humerus_right', 'ulna_effector_right', 'ulna_right', 'radius_effector_right',
            'radius_right', 'hand_right']
        - muscle: string, name of the muscle.
        - segment_index: (default = None) int, index position of the segment in the segment list of the model.
        - gcs_seg_0: (default = None) 4x4 array, rotation matrix and translation vector of the segment in the initial 
        configuration.
        """
        
        self.radius = radius
        self.side = side # -1 or 1
        self.c1_initial = c1 
        self.c2_initial = c2
        self.c1 = c1
        self.c2 = c2
        self.matrix_initial = matrix
        self.matrix = matrix
        self.point_on_cylinder = point_on_cylinder
        self.segment = segment
        self.muscle = muscle
        self.segment_index = segment_index
        self.gcs_seg_0 = gcs_seg_0

    @classmethod
    def from_matrix(cls, radius, side, matrix, point_on_cylinder=False, segment=None, muscle = None, d = 0.1):
        
        """ Creates a cylinder with a given transformation matrix.
        
        Args: 
        - radius: float, radius of the cylinder.
        - side: int (-1 or 1) indicating the handedness of the side:
            - 1 for the right-handed side
            - -1 for the left-handed side (with respect to the z-axis)
        - matrix: 4x4 array, rotation matrix and translation vector.
        - point_on_cylinder: bool (default is False). Set to True if the origin or insertion point is on the surface of 
        the cylinder. 
        Warning: This option may require adjustments for proper functionality. 
        It was originally introduced to prevent the point from entering the cylinder and to adjust the cylinder's 
        radius so that the point remains on the surface. This is particularly useful for PECM2 and PECM3 with 
        two wrapped cylinders. Ensure the correct order of cylinders in the "cylinders" list:
        - Example:
            - cylinders = [cylinder_1, cylinder_2]
            - cylinder_1.point_on_cylinder = True  # Origin point is on the surface
            - cylinder_2.point_on_cylinder = True  # Insertion point is on the surface
        - segment: (default = None) The segment of the cylinder. Choose an authorized name from the following list:
            ['thorax', 'spine', 'clavicle_effector_right', 'clavicle_right', 'scapula_effector_right',
            'scapula_right', 'humerus_right', 'ulna_effector_right', 'ulna_right', 'radius_effector_right',
            'radius_right', 'hand_right']
        - muscle: string, name of the muscle.
        - d: scalar distance to calculate the endpoints of the cylinder.
        """
        
        # Extract the origin point and z-direction vector from the transformation matrix
        origin = matrix[0:-1, -1]
        z = matrix[2, 0:-1]
        
        # Calculate the norm (length) of the vector from origin to z
        norm_AB = np.linalg.norm(z - origin)
        # Calculate the unit vector in the direction from origin to z
        unit_AB = (z - origin) / norm_AB
        
        # Calculate the coordinates of centers of the cylinder
        c1 = origin - d * unit_AB
        c2 = origin + d * unit_AB
        
        return cls(radius, side, c1, c2, matrix, point_on_cylinder, segment, muscle)

    @classmethod
    def from_points(cls, radius, side, c1, c2, point_on_cylinder=False, segment=None, muscle=None):
        """Create a cylinder with two points and create a matrix for this cylinder
        
        Args: 
        - radius: float, radius of the cylinder.
        - side: int (-1 or 1) indicating the handedness of the side:
            - 1 for the right-handed side
            - -1 for the left-handed side (with respect to the z-axis)
        - c1_initial: 3x1 array, coordinates of the center of the first circle (bottom or top) in the initial 
        configuration (q_initial = 0).
        - c2_initial: 3x1 array, coordinates of the center of the second circle (top or bottom) in the initial 
        configuration (q_initial = 0).
        - point_on_cylinder: bool (default is False). Set to True if the origin or insertion point is on the surface of 
        the cylinder. 
        Warning: This option may require adjustments for proper functionality. 
        It was originally introduced to prevent the point from entering the cylinder and to adjust the cylinder's 
        radius so that the point remains on the surface. This is particularly useful for PECM2 and PECM3 with 
        two wrapped cylinders. Ensure the correct order of cylinders in the "cylinders" list:
        - Example:
            - cylinders = [cylinder_1, cylinder_2]
            - cylinder_1.point_on_cylinder = True  # Origin point is on the surface
            - cylinder_2.point_on_cylinder = True  # Insertion point is on the surface
        - segment: (default = None) The segment of the cylinder. Choose an authorized name from the following list:
            ['thorax', 'spine', 'clavicle_effector_right', 'clavicle_right', 'scapula_effector_right',
            'scapula_right', 'humerus_right', 'ulna_effector_right', 'ulna_right', 'radius_effector_right',
            'radius_right', 'hand_right']
        - muscle: string, name of the muscle.
        """
        
        # Find the frame of the cylinder based on the two centers
        frame = find_cylinder_frame([c1, c2])
        # Determine the transformation matrix based on the cylinder's frame and midpoint
        midpoint = (c1 + c2) / 2
        matrix = find_matrix(frame, midpoint)
        
        return cls(radius, side, c1, c2, matrix, point_on_cylinder, segment, muscle)
        
    def rotate_around_axis(self, alpha) :
        """
        Rotates the cylinder around its axis by a given angle.

        Args:
        - alpha: float, angle in degrees to rotate the cylinder around its axis.

        Updates:
        - Updates the cylinder's transformation matrix with the new rotation.
        """
         
        alpha = np.radians(alpha) # Convert the angle from degrees to radians
        # Create a 3x3 rotation matrix for rotation around the z-axis
        rotation_axe_cylinder = np.array([[np.cos(alpha), - np.sin(alpha), 0],
                                        [ np.sin(alpha), np.cos(alpha), 0],
                                        [ 0.       ,  0.        ,  1.    ]])
        # Apply the rotation matrix to the current rotation part of the transformation matrix
        new_rotation_matrix = np.dot(self.matrix[:3, :3],(rotation_axe_cylinder))
        
        # Update the full 4x4 transformation matrix with the new rotation matrix
        self.matrix = np.array([[ new_rotation_matrix[0,0], new_rotation_matrix[0,1],new_rotation_matrix[0,2], self.matrix[0,3]],
                                [ new_rotation_matrix[1,0], new_rotation_matrix[1,1],new_rotation_matrix[1,2], self.matrix[1,3]],
                                [ new_rotation_matrix[2,0], new_rotation_matrix[2,1],new_rotation_matrix[2,2], self.matrix[2,3]],
                                 [ 0.        ,  0.        ,  0.        ,  1.        ]])

    def compute_new_radius(self, coordinates_points_on_cylinder) : 
        """ Adjusts the cylinder's radius so that a given point remains on its surface.

        Warning:
        Some adjustments may be necessary for proper functionality. This function is designed to prevent the point 
        from entering the cylinder and to ensure that the cylinder's radius is adjusted so that the point stays on 
        its surface. This is particularly useful for PECM2 and PECM3 when wrapping two cylinders. Ensure the correct 
        order of cylinders in the "cylinders" list:
        - Example:
        - cylinders = [cylinder_1, cylinder_2]
        - cylinder_1.point_on_cylinder = True  # Origin point is on the surface
        - cylinder_2.point_on_cylinder = True  # Insertion point is on the surface

        Args:
        - coordinates_points_on_cylinder: Array of coordinates of points that should be on the surface of the cylinder.
        This array should be in the global frame of reference.

        Updates:
        - Adjusts the `self.radius` attribute of the cylinder based on the provided coordinates.
        """
        
        if self.point_on_cylinder == True : 
            # Transform the coordinates from the global frame to the cylinder's local frame
            coordinates_points_on_cylinder_local = transpose_switch_frame(coordinates_points_on_cylinder, self.matrix)
            # Calculate the new radius based on the transformed coordinates
            self.radius = np.linalg.norm(coordinates_points_on_cylinder_local[:2])
            # print("radius = ", self.radius)
        
    def compute_new_matrix_segment(self, model, q) :
        
        """ Computes the transformation matrix for the segment based on new joint configurations.

        Args:
        - model: The model object containing the global joint coordinate systems (JCS).
        - q: A 4x2 array representing the new configuration for the humerus segment.

        Updates:
        - self.matrix: The updated transformation matrix for the segment.
        - self.c1 and self.c2: Updated coordinates of the cylinder's end circles in the new frame.
        
        """
        # Retrieve the global JCS for the segment based on the new configuration
        gcs_seg = [gcs.to_array() for gcs in model.allGlobalJCS(q)][self.segment_index]
        
        # Compute the rotation matrix to align the initial GCS with the new GCS
        rot = self.gcs_seg_0[:3, :3].T  
        rototrans = np.eye(4)
        rototrans[:3, :3] = rot
        rototrans[:3, 3] = -rot @ self.gcs_seg_0[:3, 3]
        
        # Transform the initial cylinder end circles to the new segment frame
        self.c1, self.c2 = transpose_switch_frame([self.c1_initial, self.c2_initial], self.gcs_seg_0)
        # c1 and c2 are now in the local frame of the humerus 
        
        # Transform the points to the global frame with the new segment GCS
        self.c1, self.c2 = switch_frame([self.c1, self.c2], gcs_seg)
        # Now c1 and c2 are in the global frame with the updated GCS
        
        # Compute the cylinder frame and the transformation matrix
        frame = find_cylinder_frame([self.c1, self.c2])
        midpoint = (self.c1 + self.c2) / 2
        self.matrix = find_matrix(frame, midpoint)
        
        
    def compute_matrix_rotation_zy(self, matrix_rot_zy):
        """ Applies a rotation transformation from the z-axis to the y-axis to the current transformation matrix.

        Args:
        - matrix_rot_zy: A 4x4 rotation matrix that represents the rotation from the z-axis to the y-axis.

        Updates:
        - Modifies the instance's transformation matrix by applying the rotation transformation.
        """
        # Apply the rotation matrix to the current transformation matrix
        self.matrix = np.dot(matrix_rot_zy, self.matrix)

    def compute_seg_index_and_gcs_seg_0(self, q_initial, model, segment_names):
        """
        Computes the initial global coordinate system (GCS) for a specified segment and its index.

        Args:
        - q_initial: A 4x2 array representing the initial configuration of the humerus segment.
        - model: The model object that contains segment information and global coordinate systems.
        - segment_names: A list of all segment names in the model.

        Updates:
        - self.segment_index: Index of the segment in the `segment_names` list.
        - self.gcs_seg_0: The initial global coordinate system (GCS) matrix for the specified segment.
        """
        # Compute the index of the specified segment in the list of segment names
        self.segment_index = segment_names.index(self.segment)
        
        # Compute the global coordinate systems (GCS) for all segments with respect to q_initial
        all_gcs = [gcs.to_array() for gcs in model.allGlobalJCS(q_initial)]
        
        # Retrieve the GCS of the specified segment using its index
        self.gcs_seg_0 = all_gcs[self.segment_index]
        
    def change_side(self):
        """
        Flips the side of the cylinder by changing its sign.
        
        The side is multiplied by -1, effectively toggling between the right-handed
        and left-handed side definitions of the cylinder.
        """
        self.side *= -1

    # def rotation_direction(self, matrix_initial, matrix_update):
    #     """Determine the rotation direction around the z-axis between two matrices.

    #     Args:
    #     - matrix_initial: 4x4 initial rotation-translation matrix
    #     - matrix_update: 4x4 updated rotation-translation matrix

    #     Returns:
    #     - 1 if the rotation is clockwise
    #     - -1 if the rotation is counterclockwise
    #     """
    #     # Extract the rotation angles from both matrices
    #     _, _, theta_z_initial = self.extract_angles(matrix_initial)
    #     _, _, theta_z_update = self.extract_angles(matrix_update)

    #     # Calculate the change in the z rotation angle
    #     delta_theta_z = theta_z_update - theta_z_initial

    #     # Normalize the delta_theta_z to be within -pi to pi
    #     delta_theta_z = (delta_theta_z + np.pi) % (2 * np.pi) - np.pi

    #     # Determine the direction
    #     if delta_theta_z > 0:
    #         direction = 1
    #     else:
    #         direction = -1

    #     # If the absolute change is greater than pi, reverse the direction
    #     if abs(delta_theta_z) > np.pi:
    #         direction *= -1

    #     return direction

    def compute_if_tangent_point_in_cylinder(self, p1, p2, bool_inactive):
        """
        Determines if either of the tangent points is within the cylinder or if the segment intersects with the cylinder.

        Args:
        - p1: Coordinates of the first tangent point in the global frame.
        - p2: Coordinates of the second tangent point in the global frame.
        - bool_inactive: Boolean flag indicating if the segment's intersection with the cylinder should be checked.

        Returns:
        - True if either tangent point is inside the cylinder or if the segment intersects with the cylinder (when `bool_inactive` is True).
        - False otherwise.
        """
        
        # Transform the points into the cylinder's local frame
        p1_local, p2_local = transpose_switch_frame([p1, p2], self.matrix)
        
        # Compute the distances from the origin to the points in the local frame
        r1 = np.linalg.norm(p1_local[:2])
        r2 = np.linalg.norm(p2_local[:2])
        
        # Check if either point is inside the cylinder or if the segment intersects with the cylinder
        if (r1 < self.radius or r2 < self.radius) or (
            bool_inactive and does_segment_intersect_cylinder(p1_local, p2_local, self.radius)):
            return True
        else:
            return False

    def __str__(self):
        return (f"Cylinder(radius = {self.radius}, "
                f"side=\n{self.side}, "
                f"matrix_initial=\n{self.matrix_initial}, "
                f"matrix=\n{self.matrix}, "
                f"segment : {self.segment},"
                f"segment_index=\n{self.segment_index}, "
                f"gcs_seg_0=\n{self.gcs_seg_0})")
        
# ----------

def find_cylinder_frame(center_circle):
    """
    Constructs a 3x3 orthonormal frame for a cylinder based on the coordinates of its two end circles.

    Args:
    - center_circle: 2x3 array, the coordinates of the first and second circles of the cylinder.

    Returns:
    - cylinder_frame: 3x3 numpy array, orthonormal frame for the cylinder.

    """
    # Compute the vector from the first to the second circle, which defines the cylinder's z-axis
    vect = center_circle[1] - center_circle[0]
    unit_vect = vect / np.linalg.norm(vect) # Normalize the vector to get the z-axis direction

    # Select a vector not parallel to the z-axis for the initial y-axis direction
    not_unit_vect = np.array([1, 0, 0])
    if np.all(unit_vect == [1, 0, 0]) or np.all(unit_vect == [-1, 0, 0]):
        not_unit_vect = np.array([0, 1, 0])

    # Compute a normalized vector perpendicular to the z-axis to use as the y-axis
    n1 = np.cross(unit_vect, not_unit_vect)
    n1 = n1 / np.linalg.norm(n1) # Normalize to get the y-axis direction

    # Compute the x-axis as the cross product of y and z axes
    n2 = np.cross(n1, unit_vect) # x-axis direction

    # Return the orthonormal frame as a 3x3 matrix with columns [x, y, z]
    return np.transpose(np.array([n2, n1, unit_vect]))

def find_matrix(cylinder_frame, origin):
    """
    Constructs a 4x4 transformation matrix from a given frame and origin.

    Args:
    - cylinder_frame: 3x3 matrix, rotation (orientation) of the cylinder frame.
    - origin: 3-element array, translation vector (origin) of the cylinder frame.

    Returns:
    - 4x4 numpy array, transformation matrix with rotation and translation.

    Note:
    The resulting matrix is of the form:
    [ R | T ]
    [ 0 | 1 ]
    where R is the 3x3 rotation matrix and T is the translation vector.
    """
    return np.array([
        [cylinder_frame[0][0], cylinder_frame[0][1], cylinder_frame[0][2], origin[0]],
        [cylinder_frame[1][0], cylinder_frame[1][1], cylinder_frame[1][2], origin[1]],
        [cylinder_frame[2][0], cylinder_frame[2][1], cylinder_frame[2][2], origin[2]],
        [0, 0, 0, 1]
    ])

def does_segment_intersect_cylinder(P1, P2, R, epsilon=0.0001):
    """
    Determines if a line segment intersects with a cylinder when projected onto the XY plane.
    The function projects the line segment onto the XY plane and checks if the segment intersects
    with the circle representing the cylinder's cross-section. It solves a quadratic equation to
    determine intersection points and checks if they lie within the segment.

    Args:
    - P1: Coordinates of the first endpoint of the segment (x1, y1, z1).
    - P2: Coordinates of the second endpoint of the segment (x2, y2, z2).
    - R: Radius of the cylinder.
    - epsilon: Small value to adjust the radius for numerical stability (default is 0.0001).

    Returns:
    - True if the segment intersects with the cylinder, False otherwise.
    """
    R -= epsilon  # Adjust the radius for numerical stability

    # Extract x and y coordinates from points P1 and P2
    x1, y1, z1 = P1
    x2, y2, z2 = P2
    
    # Calculate the coefficients of the quadratic equation for the segment's projection onto the XY plane
    dx = x2 - x1
    dy = y2 - y1
    a = dx**2 + dy**2
    b = 2 * (x1 * dx + y1 * dy)
    c = x1**2 + y1**2 - R**2
    
    # Solve the quadratic equation a*t^2 + b*t + c = 0 to find intersection points
    discriminant = b**2 - 4 * a * c
    
    if discriminant < 0:
        # No intersection with the cylinder
        return False
    
    # Calculate the roots of the quadratic equation
    sqrt_discriminant = np.sqrt(discriminant)
    t1 = (-b + sqrt_discriminant) / (2 * a)
    t2 = (-b - sqrt_discriminant) / (2 * a)
    
    # Check if either root lies within the segment interval [0, 1]
    if (0 <= t1 <= 1) or (0 <= t2 <= 1):
        return True
    else:
        return False
    
# def extract_angles(self, matrix):
#     """
#     Extracts rotation angles around the x, y, and z axes from a 4x4 rotation-translation matrix.

#     Args:
#     - matrix: A 4x4 transformation matrix in the form:
#     [ R | T ]
#     [ 0 | 1 ]
#     where R is the 3x3 rotation matrix and T is the translation vector.

#     Returns:
#     - theta_x: Rotation angle around the x-axis in radians.
#     - theta_y: Rotation angle around the y-axis in radians.
#     - theta_z: Rotation angle around the z-axis in radians.

#     Note:
#     This function assumes that the rotation matrix R is orthonormal and represents a rotation
#     matrix without scaling or skewing.
#     """
#     # Extract the 3x3 rotation matrix from the 4x4 matrix
#     R = matrix[:3, :3]

#     # Compute the rotation angles using atan2 for better numerical stability
#     theta_x = np.arctan2(R[2, 1], R[2, 2])
#     theta_y = np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2))
#     theta_z = np.arctan2(R[1, 0], R[0, 0])

#     return theta_x, theta_y, theta_z