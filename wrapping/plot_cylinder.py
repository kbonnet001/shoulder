import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import norm
from wrapping.step_1 import switch_frame, transpose_switch_frame

def create_cylinder(radius, height, num_points=100):
    """
    Create a grid representation of a cylinder.

    Args:
        radius (float): Radius of the cylinder.
        height (float): Height of the cylinder.
        num_points (int, optional): Number of points to use for representation. Default is 100.

    Returns:
        x_grid (ndarray): Array of x-coordinates for the cylinder surface.
        y_grid (ndarray): Array of y-coordinates for the cylinder surface.
        z_grid (ndarray): Array of z-coordinates for the cylinder surface.
    """
    # Generate linearly spaced values for the z-coordinate from -height/2 to height/2
    z = np.linspace(-height / 2, height / 2, num_points)
    
    # Generate linearly spaced values for the angular coordinate (theta) from 0 to 2*pi
    theta = np.linspace(0, 2 * np.pi, num_points)
    
    # Create a meshgrid for theta and z to cover the surface of the cylinder
    theta_grid, z_grid = np.meshgrid(theta, z)
    
    # Compute the x and y coordinates based on the radius and angular coordinate (theta)
    x_grid = radius * np.cos(theta_grid)
    y_grid = radius * np.sin(theta_grid)
    
    # Return the grids of x, y, and z coordinates
    return x_grid, y_grid, z_grid

def apply_transformation(Cylinder, height, num_points=100):
    """
    Apply a transformation matrix to a cylinder.

    Args:
        Cylinder (object): An object with a `matrix` attribute, which is a 4x4 transformation matrix.
        height (float): Height of the cylinder.
        num_points (int, optional): Number of points to use for representation. Default is 100.

    Returns:
        x_transformed (ndarray): Array of x-coordinates for the transformed cylinder surface.
        y_transformed (ndarray): Array of y-coordinates for the transformed cylinder surface.
        z_transformed (ndarray): Array of z-coordinates for the transformed cylinder surface.
    """
    # Generate the grid representation of the cylinder
    x, y, z = create_cylinder(Cylinder.radius, height, num_points)
    
    # Get the shape of the grid arrays
    shape = x.shape
    
    # Create an array of ones to be used for homogeneous coordinates
    ones = np.ones(shape[0] * shape[1])
    
    # Stack the x, y, z coordinates and ones into a single array for homogeneous transformation
    points = np.vstack((x.flatten(), y.flatten(), z.flatten(), ones))
    
    # Apply the transformation matrix to the points
    transformed_points = Cylinder.matrix @ points
    
    # Reshape the transformed coordinates back to the original grid shape
    x_transformed = transformed_points[0].reshape(shape)
    y_transformed = transformed_points[1].reshape(shape)
    z_transformed = transformed_points[2].reshape(shape)
    
    return x_transformed, y_transformed, z_transformed

def data_cylinder(center_circle_1, center_circle_2, cylinder_frame, radius, num_points=100):
    """
    Compute data for plotting a cylinder.

    Args:
        center_circle_1 (ndarray): Coordinates of the first circle of the cylinder (3x1 array).
        center_circle_2 (ndarray): Coordinates of the second circle of the cylinder (3x1 array).
        cylinder_frame (ndarray): Local frame of the cylinder (3x3 array).
        radius (float): Radius of the cylinder.
        num_points (int, optional): Number of points for the representation. Default is 100.

    Returns:
        X (ndarray): X-coordinates of the cylinder surface.
        Y (ndarray): Y-coordinates of the cylinder surface.
        Z (ndarray): Z-coordinates of the cylinder surface.
    """
    # Compute the direction vector of the cylinder's axis
    v_cylinder = center_circle_2 - center_circle_1

    # Unpack cylinder frame vectors
    n2, n1, v_unit = cylinder_frame

    # Create a grid of points along the cylinder's axis and around its circumference
    t = np.linspace(0, norm(v_cylinder), num_points)
    theta = np.linspace(0, 2 * np.pi, num_points)
    t, theta = np.meshgrid(t, theta)

    # Calculate coordinates for the cylinder surface
    X = center_circle_1[0] + v_unit[0] * t + radius * np.sin(theta) * n1[0] + radius * np.cos(theta) * n2[0]
    Y = center_circle_1[1] + v_unit[1] * t + radius * np.sin(theta) * n1[1] + radius * np.cos(theta) * n2[1]
    Z = center_circle_1[2] + v_unit[2] * t + radius * np.sin(theta) * n1[2] + radius * np.cos(theta) * n2[2]

    return X, Y, Z

def data_semi_circle(v1, v2, matrix, r, num_points=100):
    """
    Compute data for plotting a semi-circle between two tangent points.

    Args:
        v1 (ndarray): Coordinates of the first tangent point (3x1 array).
        v2 (ndarray): Coordinates of the second tangent point (3x1 array).
        matrix (ndarray): Transformation matrix (4x4 array).
        r (float): Radius of the semi-circle.
        num_points (int, optional): Number of points for the representation. Default is 100.

    Returns:
        semi_circle_points (ndarray): Coordinates of points representing the semi-circle.
    """
    # Transform tangent points to the desired frame
    v1, v2 = transpose_switch_frame([v1, v2], matrix)
    c = np.array([0, 0, (v1[2] + v2[2]) / 2])

    # Compute the normal vector to the plane defined by v1, v2, and c
    norm = np.cross(v1 - c, v2 - c)
    norm /= np.linalg.norm(norm)

    # Calculate the angle between v1 and v2
    angle = np.arccos(np.dot((v1 - c) / np.linalg.norm(v1 - c), (v2 - c) / np.linalg.norm(v2 - c)))

    # Generate points for the semi-circle
    theta = np.linspace(0, angle, num_points)
    semi_circle_points = c + r * np.cos(theta)[:, np.newaxis] * (v1 - c) / np.linalg.norm(v1 - c) + \
                          r * np.sin(theta)[:, np.newaxis] * np.cross(norm, (v1 - c) / np.linalg.norm(v1 - c))

    # Transform points back to the original frame
    for i in range(len(semi_circle_points)):
        semi_circle_points[i] = switch_frame(semi_circle_points[i], matrix)

    return semi_circle_points

def plot_one_cylinder_obstacle(origin_point, final_point, Cylinder, v1, v2, obstacle_tangent_point_inactive):
    """
    Plot the representation of a single-cylinder obstacle with the muscle path.

    Args:
        origin_point (ndarray): Position of the first point (3x1 array).
        final_point (ndarray): Position of the second point (3x1 array).
        Cylinder (object): Cylinder object containing radius and transformation matrix.
        v1 (ndarray): Position of the first obstacle tangent point (3x1 array).
        v2 (ndarray): Position of the second obstacle tangent point (3x1 array).
        obstacle_tangent_point_inactive (bool): True if v1 and v2 are inactive; False otherwise.
    
    Returns:
        None: Displays the plot.
    """
    # Create figure and 3D axis
    fig = plt.figure("Single Cylinder Wrapping")
    ax = fig.add_subplot(111, projection='3d')

    # Plot origin and final points
    ax.scatter(*origin_point, color='g', label="Origin point")
    ax.scatter(*final_point, color='b', label="Final point")

    # Plot obstacle tangent points
    ax.scatter(*v1, color='r', label="v1")
    ax.scatter(*v2, color='r', label="v2")

    # Plot the cylinder
    Xc, Yc, Zc = apply_transformation(Cylinder, 0.2, 100)
    ax.plot_surface(Xc, Yc, Zc, alpha=0.5)

    if obstacle_tangent_point_inactive:
        # Plot straight line path if tangent points are inactive
        ax.plot(*zip(origin_point, final_point), color='r')
    else:
        # Plot semi-circle path if tangent points are active
        semi_circle_points = data_semi_circle(v1, v2, Cylinder.matrix, Cylinder.radius, 100)
        ax.plot(semi_circle_points[:, 0], semi_circle_points[:, 1], semi_circle_points[:, 2])

        # Plot the muscle path segments
        ax.plot(*zip(origin_point, v1), color='g')
        ax.plot(*zip(v2, final_point), color='b')

    # Set axis labels
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.grid(True)

    # Set axis limits
    ax.set_ylim(-0.1, 0.1)
    ax.set_zlim(-0.2, 0)
    ax.set_xlim(0, 0.2)

    plt.title("Single Cylinder Wrapping")
    plt.legend()
    plt.show()

def plot_double_cylinder_obstacle(P, S, Cylinder_U, Cylinder_V, Q, G, H, T, Q_G_inactive, H_T_inactive):
    """
    Plot the representation of a double-cylinder obstacle with the muscle path.

    Args:
        P (ndarray): Position of the first point (3x1 array).
        S (ndarray): Position of the second point (3x1 array).
        Cylinder_U (object): Cylinder U object containing radius and transformation matrix.
        Cylinder_V (object): Cylinder V object containing radius and transformation matrix.
        Q (ndarray): Position of the first obstacle tangent point (3x1 array).
        G (ndarray): Position of the second obstacle tangent point (3x1 array).
        H (ndarray): Position of the third obstacle tangent point (3x1 array).
        T (ndarray): Position of the fourth obstacle tangent point (3x1 array).
        Q_G_inactive (bool): True if Q and G are inactive; False otherwise.
        H_T_inactive (bool): True if H and T are inactive; False otherwise.
    
    Returns:
        None: Displays the plot.
    """
    # Create figure and 3D axis
    fig = plt.figure("Double Cylinder Wrapping")
    ax = fig.add_subplot(111, projection='3d')

    # Plot points
    ax.scatter(*P, color='g', label="Origin point")
    ax.scatter(*S, color='b', label="Final point")
    ax.scatter(*Q, color='r', label="Q")
    ax.scatter(*G, color='r', label="G")
    ax.scatter(*H, color='r', label="H")
    ax.scatter(*T, color='r', label="T")

    # Plot Cylinder U
    Xcu, Ycu, Zcu = apply_transformation(Cylinder_U, 0.2, 100)  # Height 0.2
    ax.plot_surface(Xcu, Ycu, Zcu, alpha=0.5)

    # Plot Cylinder V
    Xcv, Ycv, Zcv = apply_transformation(Cylinder_V, 0.2, 100)  # Height 0.2
    ax.plot_surface(Xcv, Ycv, Zcv, alpha=0.5)

    if Q_G_inactive and H_T_inactive:
        # Muscle path is a straight line from P to S
        ax.plot(*zip(P, S), color='r')

    elif Q_G_inactive:
        # Single-cylinder algorithm with Cylinder V
        semi_circle_points = data_semi_circle(H, T, Cylinder_V.matrix, Cylinder_V.radius, 100)
        ax.plot(semi_circle_points[:, 0], semi_circle_points[:, 1], semi_circle_points[:, 2])
        ax.plot(*zip(P, H), color='g')
        ax.plot(*zip(T, S), color='b')

    elif H_T_inactive:
        # Single-cylinder algorithm with Cylinder U
        semi_circle_points = data_semi_circle(Q, G, Cylinder_U.matrix, Cylinder_U.radius, 100)
        ax.plot(semi_circle_points[:, 0], semi_circle_points[:, 1], semi_circle_points[:, 2])
        ax.plot(*zip(P, Q), color='g')
        ax.plot(*zip(G, S), color='b')

    else:
        # Double-cylinder algorithm
        semi_circle_points_U = data_semi_circle(Q, G, Cylinder_U.matrix, Cylinder_U.radius, 100)
        ax.plot(semi_circle_points_U[:, 0], semi_circle_points_U[:, 1], semi_circle_points_U[:, 2])
        semi_circle_points_V = data_semi_circle(H, T, Cylinder_V.matrix, Cylinder_V.radius, 100)
        ax.plot(semi_circle_points_V[:, 0], semi_circle_points_V[:, 1], semi_circle_points_V[:, 2])
        ax.plot(*zip(P, Q), color='g')
        ax.plot(*zip(G, H), color='b')
        ax.plot(*zip(T, S), color='b')

    # Set axis labels and limits
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.grid(True)
    ax.set_ylim(-0.1, 0.1)
    ax.set_zlim(-0.2, 0)
    ax.set_xlim(0, 0.2)

    plt.title("Double Cylinder Wrapping")
    plt.legend()
    plt.show()

def plot_cadran_single_cylinder(P, S, Cylinder, p1, p2, points_tangent_inactif):
    """
    Plot the muscle path and interaction with a single cylinder.

    Args:
        P (array): 3x1 array representing the position of the origin point.
        S (array): 3x1 array representing the position of the insertion point.
        Cylinder (object): Cylinder object containing properties like radius and segment.
        p1 (array): 3x1 array representing the position of the first tangent point.
        p2 (array): 3x1 array representing the position of the second tangent point.
        points_tangent_inactif (bool): Indicates if the tangent points are inactive.

    """
    # Create a new figure and axis for the plot
    _, ax = plt.subplots()
    ax.grid(True)  

    # Plot the origin point, insertion point, and tangent points
    ax.plot(P[0], P[1], 'go', label='origin')  # Plot origin point in green
    ax.plot(S[0], S[1], 'bo', label='insertion')  # Plot insertion point in blue
    ax.plot(p1[0], p1[1], 'mo', label='p1')  # Plot first tangent point in magenta
    ax.plot(p2[0], p2[1], 'ro', label='p2')  # Plot second tangent point in red
    
    # If tangent points are active, plot lines between points and semi-circle
    if not points_tangent_inactif:
        ax.plot(*zip(P, p1), color='g')  # Draw line from origin to first tangent point
        ax.plot(*zip(p2, S), color='b')  # Draw line from second tangent point to insertion point
        # Generate and plot the semi-circle representing the wrapping around the cylinder
        semi_circle_points = data_semi_circle(np.concatenate((p1, [1])), np.concatenate((p2, [1])), np.eye(4), Cylinder.radius, 100)
        ax.plot(semi_circle_points[:, 0], semi_circle_points[:, 1], color="r")
    else:
        # If tangent points are inactive, plot the direct path from origin to insertion
        ax.plot(*zip(P, S), color='r')

    # Draw the cylinder as a dashed circle
    circle = plt.Circle([0, 0], Cylinder.radius, color='k', fill=False, linestyle='--', linewidth=2, label=f'Cylinder')
    ax.add_artist(circle)

    # Set labels for the axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
    # Set the limits for the x and y axes to ensure all elements are visible
    ax.set_xlim(min(P[0], S[0], p1[0], p2[0]) - Cylinder.radius - 0.01, 
                max(P[0], S[0], p1[0], p2[0]) + Cylinder.radius + 0.01)
    ax.set_ylim(min(P[1], S[1], p1[1], p2[1]) - Cylinder.radius - 0.01, 
                max(P[1], S[1], p1[1], p2[1]) + Cylinder.radius + 0.01)
    ax.legend()

    # Set the title of the plot
    plt.title(f"Cylinder {Cylinder.segment} local frame")
    plt.show()  # Display the plot

def plot_cadran_double_cylinder(P, S, Cylinders, p1, p2, points_tangent_inactif, names_cylinder=["", ""]):
    """
    Plot the muscle path and interactions with two cylinders.

    Args:
        P (list of arrays): List of 3x1 arrays representing the positions of the origin points for both cylinders.
        S (list of arrays): List of 3x1 arrays representing the positions of the insertion points for both cylinders.
        Cylinders (list of objects): List of Cylinder objects containing properties like radius and segment for both cylinders.
        p1 (list of arrays): List of 3x1 arrays representing the positions of the first tangent points for both cylinders.
        p2 (list of arrays): List of 3x1 arrays representing the positions of the second tangent points for both cylinders.
        points_tangent_inactif (list of bools): List indicating if the tangent points are inactive for each cylinder.
        names_cylinder (list of strings, optional): List of names or identifiers for the cylinders, for labeling.

    """
    # Create a new figure with two subplots side by side
    _, axs = plt.subplots(1, 2, figsize=(12, 6))
    
    # Iterate over the two subplots and corresponding cylinder data
    for k in range(2):
        ax = axs[k]
        ax.grid(True) 

        # Plot the origin, insertion, and tangent points for the current cylinder
        ax.plot(P[k][0], P[k][1], 'go', label='Origin')  # Plot origin point in green
        ax.plot(S[k][0], S[k][1], 'bo', label='Insertion')  # Plot insertion point in blue
        ax.plot(p1[k][0], p1[k][1], 'mo', label='p1')  # Plot first tangent point in magenta
        ax.plot(p2[k][0], p2[k][1], 'ro', label='p2')  # Plot second tangent point in red
        
        # Check if tangent points are active or inactive
        if not points_tangent_inactif[k]:
            # If active, plot lines between points and semi-circle
            ax.plot(*zip(P[k], p1[k]), color='g')  # Line from origin to first tangent point
            ax.plot(*zip(p2[k], S[k]), color='b')  # Line from second tangent point to insertion point
            # Generate and plot the semi-circle representing the wrapping around the cylinder
            semi_circle_points = data_semi_circle(np.concatenate((p1[k], [1])), np.concatenate((p2[k], [1])), np.eye(4), Cylinders[k].radius, 100)
            ax.plot(semi_circle_points[:, 0], semi_circle_points[:, 1], color="r")
        else:
            # If inactive, plot the direct path from origin to insertion
            ax.plot(*zip(P[k], S[k]), color='r')

        # Draw the cylinder as a dashed circle
        circle = plt.Circle((0, 0), Cylinders[k].radius, color='k', fill=False, linestyle='--', linewidth=2, label='Cylinder')
        ax.add_artist(circle)

        # Set labels for the axes
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        # Adjust axis limits to ensure all elements are visible
        ax.set_xlim(min(P[k][0], S[k][0], p1[k][0], p2[k][0]) - Cylinders[k].radius - 0.01, 
                    max(P[k][0], S[k][0], p1[k][0], p2[k][0]) + Cylinders[k].radius + 0.01)
        ax.set_ylim(min(P[k][1], S[k][1], p1[k][1], p2[k][1]) - Cylinders[k].radius - 0.01, 
                    max(P[k][1], S[k][1], p1[k][1], p2[k][1]) + Cylinders[k].radius + 0.01)
        ax.legend()  # Add a legend to the plot
        ax.set_title(f"Cylinder {Cylinders[k].segment} local frame") # Set the title for each subplot

    # Adjust layout to prevent overlap and display the plot
    plt.tight_layout()
    plt.show()

    




