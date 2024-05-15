import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import norm
from wrapping.plot_cylinder import *
from wrapping.algorithm import*
from wrapping.step_1 import find_cylinder_frame

# ####################
# import biorbd
# import bioviz

# model = biorbd.Model("models/Wu_DeGroote.bioMod")
# q = np.zeros((model.nbQ(), ))
# q[1] = -2

# segment_names = [model.segment(i).name().to_string() for i in range(model.nbSegment())]
# humerus_index = segment_names.index("humerus_right")


# humerus_dof_names = [model.segment(humerus_index).nameDof(i).to_string() for i in range(model.segment(humerus_index).nbQ())]
# q_ranges = [[ranges.min(), ranges.max()] for ranges in model.segment(humerus_index).QRanges()]

# q[1] = -2
# gcs = [gcs.to_array() for gcs in model.allGlobalJCS(q)]
# gcs_humerus = gcs[humerus_index]
# # Cylinder pose
# model.updateMuscles(q)
# mus = model.muscle(0)
# origin_0 = mus.musclesPointsInGlobal(model, q)[0].to_array()
# insertion_0 = mus.musclesPointsInGlobal(model, q)[-1].to_array()

# q[1] = 0
# gcs = [gcs.to_array() for gcs in model.allGlobalJCS(q)]
# gcs_humerus = gcs[humerus_index]
# # Cylinder pose
# model.updateMuscles(q)
# mus = model.muscle(0)
# origin_1 = mus.musclesPointsInGlobal(model, q)[0].to_array()
# insertion_1 = mus.musclesPointsInGlobal(model, q)[-1].to_array()


# # transpose => [R_T, -R_T @ t; 0 0 0 1]

# # Show
# b = bioviz.Viz(loaded_model=model)
# b.set_q(q)
# b.exec()

# exit(0)
# #################

def main():

   # Provide the length wrapping around a cylinder
   #  Based on:
   #  -B.A. Garner and M.G. Pandy, The obstacle-set method for
   #  representing muscle paths in musculoskeletal models,
   #  Comput. Methods Biomech. Biomed. Engin. 3 (2000), pp. 1–30.
   # -----------------------------------------------------------

   # ------
   # Inputs
   # ------
   # Points
   P = [0,-4,-2] # origin_point
   S =[0,4,2] # final_point

   # Points for 1st cylinder
   center_circle_U = [np.array([2,0,0]),np.array([-2,0,0])]
   radius_U = 1
   side_U = 1 # Choose 1 for the right-handed side, -1 for the left-handed side (with respect to the z-axis)

   # Points for 2nd cylinder
   center_circle_V = [np.array([0,1.5,-4]),np.array([0,1.5,4])]
   radius_V = 1
   side_V = 1 # Choose 1 for the right-handed side, -1 for the left-handed side (with respect to the z-axis)

   show_details = True
   single_cylinder = False # with U cylinder
   double_cylinder = True # with U and V cylinders

   # Other inputs -----------------------------------------
   # Nothing to change he
   U_origin = (center_circle_U[1] + center_circle_U[0]) / 2
   cylinder_frame_U = find_cylinder_frame(center_circle_U)

   V_origin = (center_circle_V[1] + center_circle_V[0]) / 2
   cylinder_frame_V = find_cylinder_frame(center_circle_V)

   # Rotation matrix UV
   rotation_matrix_UV = np.dot(np.transpose(cylinder_frame_V), cylinder_frame_U)

   if (single_cylinder) :
    # --------------------------------------
    # Single cylinder obstacle set algorithm
    # --------------------------------------
    v1, v2, obstacle_tangent_point_inactive, segment_length = single_cylinder_obstacle_set_algorithm(P,S, radius_U, side_U, U_origin, cylinder_frame_U)

    # ----
    # Plot
    # ----
    plot_one_cylinder_obstacle(P,S, center_circle_U, radius_U, v1, v2, obstacle_tangent_point_inactive, segment_length, U_origin, cylinder_frame_U)

    if (show_details) :
     print("origin_point = ", P)
     print("final_point = ", S)
     print("v1 = ", v1)
     print("v2 = ", v2)
     print("obstacle_tangent_point_inactive = ",obstacle_tangent_point_inactive)
     print("segment_length = ", round(segment_length, 2))

   if (double_cylinder) :
    # --------------------------------------
    # Double cylinder obstacle set algorithm
    # --------------------------------------

    Q, G, H, T, Q_G_inactive, H_T_inactive, segment_length = double_cylinder_obstacle_set_algorithm(P, S, U_origin, cylinder_frame_U, radius_U, side_U, V_origin, cylinder_frame_V, radius_V, side_V, rotation_matrix_UV)

    # ----
    # Plot
    # ----
    plot_double_cylinder_obstacle(P, S, center_circle_U, center_circle_V, radius_U, radius_V, Q, G, H, T, cylinder_frame_U, cylinder_frame_V, U_origin, V_origin, Q_G_inactive, H_T_inactive )

    if (show_details) :
     print("origin_point = ", P)
     print("final_point = ", S)
     print("Q = ", Q)
     print("G = ", G)
     print("H = ", H)
     print("T = ", T)
     print("Q_G_inactive = ",Q_G_inactive)
     print("H_T_inactive = ",H_T_inactive)
     if (Q_G_inactive and H_T_inactive) :
      print("--> Straight line")
     elif (Q_G_inactive) :
      print("--> Single cylinder algo with V")
     elif (H_T_inactive) :
      print("--> Single cylinder algo with U")
     else :
      print("--> Double cylinder algo with U and V")
     print("segment_length = ", round(segment_length, 2))

if __name__ == "__main__":
   main()