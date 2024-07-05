# import unittest
# import numpy as np
# import biorbd
# from wrapping.step_2 import * 
# from wrapping.Cylinder import Cylinder
# from wrapping.algorithm import *
# from wrapping.plot_cylinder import *
# from wrapping.paspropre import point_to_initial, arctan2_corrected

# class Step_2_test(unittest.TestCase):
    
#     def setUp(self):
        
#         self.c1_1, self.c2_1 = np.array([2, -2, 0]), np.array([2, 2, 0]) # no wrapping
#         self.c1_2, self.c2_2 = np.array([0, -2, 0]), np.array([0, 2, 0]) # a little wrapping < pi
#         self.c1_3, self.c2_3 = np.array([-2, -1, 0]), np.array([-2, 1, 0]) # wrapping == pi
#         self.c1_4, self.c2_4 = np.array([-1.5, -0.5, 0]), np.array([-1.5, 0.5, 0]) # wrapping > pi --> arccos not work
       
#         self.cylinder_test = Cylinder.from_points(1.0, -1, np.array([0, 0, -1]), np.array([0, 0, 1]), False)

#         # algorithme single cylinder
#         # ------
#         # Step 1
#         # ------
#         self.r = self.cylinder_test.radius * self.cylinder_test.side
        
#         v1o, v2o, obstacle_tangent_point_inactive, segment_length = single_cylinder_obstacle_set_algorithm(self.c1_1, self.c2_1, self.cylinder_test, True) 
#         plot_one_cylinder_obstacle(self.c1_1, self.c2_1, self.cylinder_test, v1o, v2o, obstacle_tangent_point_inactive) 
        
#         v1o, v2o, obstacle_tangent_point_inactive, segment_length = single_cylinder_obstacle_set_algorithm(self.c1_2, self.c2_2, self.cylinder_test) 
#         # plot_one_cylinder_obstacle(self.c1_2, self.c2_2, self.cylinder_test, v1o, v2o, obstacle_tangent_point_inactive) 
#         v1o, v2o, obstacle_tangent_point_inactive, segment_length = single_cylinder_obstacle_set_algorithm(self.c1_4, self.c2_4, self.cylinder_test) 
#         # plot_one_cylinder_obstacle(self.c1_4, self.c2_4, self.cylinder_test, v1o, v2o, obstacle_tangent_point_inactive) 


#     def test_1_compute_length_v1_v2_xy(self) :
#         print("test 1")
#         P_cylinder_frame_1 = transpose_switch_frame(self.c1_1, self.cylinder_test.matrix)
#         S_cylinder_frame_1 = transpose_switch_frame(self.c2_1, self.cylinder_test.matrix)
#         v1, v2 = find_tangent_points_xy(P_cylinder_frame_1, S_cylinder_frame_1, self.r)
#         v1_v2_angle_xy = np.arccos(1.0-((v1[0]-v2[0])**2+(v1[1]-v2[1])**2)/(2*self.r**2))
#         print("v1_v2_angle_xy = ", v1_v2_angle_xy)
        
#         arctan2 = arctan2_corrected(v1, v2)
#         self.assertTrue(np.allclose(v1_v2_angle_xy, arctan2), 
#                         f"Test_1_compute_length_v1_v2_xy\nAngle must be equal to {v1_v2_angle_xy}, got: {arctan2}")
        
#         v1, v2 = find_tangent_points(P_cylinder_frame_1, S_cylinder_frame_1, self.r)
#         # ------
#         # Step 3
#         # ------
#         obstacle_tangent_point_inactive = determine_if_tangent_points_inactive_single_cylinder(v1,v2, self.r)
        
#         self.assertTrue(np.allclose(obstacle_tangent_point_inactive, True), 
#                         f"Test_1_compute_length_v1_v2_xy\nAngle must be equal to {True}, got: {obstacle_tangent_point_inactive}")
        
        
#         # ------
#         # Step 4
#         # ------
#         segment_length = segment_length_single_cylinder(obstacle_tangent_point_inactive, P_cylinder_frame_1, S_cylinder_frame_1, v1, v2, self.r)
#         print("segment length = ", segment_length)
        
#         # self.assertTrue(np.allclose(v1_v2_angle_xy, np.arctan2(v1[1], v1[0]) - np.arctan2(v2[1], v2[0])), 
#         #                 f"Test_1_compute_length_v1_v2_xy\nAngle must be equal to {np.arctan2(v1[1], v1[0]) - np.arctan2(v2[1], v2[0])}, got: {v1_v2_angle_xy}")
        
#         # self.assertTrue(np.allclose(v1_v2_angle_xy, np.arctan2(v2[1]-v1[1], v2[0]-v1[0])), 
#         #                     f"Test_1_compute_length_v1_v2_xy\nAngle must be equal to {np.arctan2(v2[1]-v1[1], v2[0]-v1[0])}, got: {v1_v2_angle_xy}")
    
#     def test_2_compute_length_v1_v2_xy(self) :
#         print("test 2")
#         P_cylinder_frame_2 = transpose_switch_frame(self.c1_2, self.cylinder_test.matrix)
#         S_cylinder_frame_2 = transpose_switch_frame(self.c2_2, self.cylinder_test.matrix)
#         v1, v2 = find_tangent_points_xy(P_cylinder_frame_2, S_cylinder_frame_2, self.r)
#         v1_v2_angle_xy = np.arccos(1.0-((v1[0]-v2[0])**2+(v1[1]-v2[1])**2)/(2*self.r**2))
#         print("v1_v2_angle_xy = ", v1_v2_angle_xy)

#         arctan2 = arctan2_corrected(v1, v2)
#         self.assertTrue(np.allclose(v1_v2_angle_xy, arctan2), 
#                         f"Test_2_compute_length_v1_v2_xy\nAngle must be equal to {v1_v2_angle_xy}, got: {arctan2}")
        
#         v1, v2 = find_tangent_points(P_cylinder_frame_2, S_cylinder_frame_2, self.r)
#         # ------
#         # Step 3
#         # ------
#         obstacle_tangent_point_inactive = determine_if_tangent_points_inactive_single_cylinder(v1,v2, self.r)
        
#         self.assertTrue(np.allclose(obstacle_tangent_point_inactive, False), 
#                         f"Test_2_compute_length_v1_v2_xy\nAngle must be equal to {False}, got: {obstacle_tangent_point_inactive}")
        
        
#         # ------
#         # Step 4
#         # ------
#         segment_length = segment_length_single_cylinder(obstacle_tangent_point_inactive, P_cylinder_frame_2, S_cylinder_frame_2, v1, v2, self.r)
#         print("segment length = ", segment_length)
        
#     def test_3_compute_length_v1_v2_xy(self) :
#         print("test 3")
#         P_cylinder_frame_3 = transpose_switch_frame(self.c1_3, self.cylinder_test.matrix)
#         S_cylinder_frame_3 = transpose_switch_frame(self.c2_3, self.cylinder_test.matrix)
#         v1, v2 = find_tangent_points_xy(P_cylinder_frame_3, S_cylinder_frame_3, self.r)
#         v1_v2_angle_xy = np.arccos(1.0-((v1[0]-v2[0])**2+(v1[1]-v2[1])**2)/(2*self.r**2))
#         print("v1_v2_angle_xy = ", v1_v2_angle_xy)
        
#         arctan2 = arctan2_corrected(v1, v2)
#         self.assertTrue(np.allclose(v1_v2_angle_xy, arctan2), 
#                         f"Test_3_compute_length_v1_v2_xy\nAngle must be equal to {v1_v2_angle_xy}, got: {arctan2}")
        
#         v1, v2 = find_tangent_points(P_cylinder_frame_3, S_cylinder_frame_3, self.r)
#         # ------
#         # Step 3
#         # ------
#         obstacle_tangent_point_inactive = determine_if_tangent_points_inactive_single_cylinder(v1,v2, self.r)
        
#         self.assertTrue(np.allclose(obstacle_tangent_point_inactive, False), 
#                         f"Test_2_compute_length_v1_v2_xy\nAngle must be equal to {False}, got: {obstacle_tangent_point_inactive}")
        
#         # ------
#         # Step 4
#         # ------
#         segment_length = segment_length_single_cylinder(obstacle_tangent_point_inactive, P_cylinder_frame_3, S_cylinder_frame_3, v1, v2, self.r)
#         print("segment length = ", segment_length)

#     def test_4_compute_length_v1_v2_xy(self) :
#         print("test 4")
#         P_cylinder_frame_4 = transpose_switch_frame(self.c1_4, self.cylinder_test.matrix)
#         S_cylinder_frame_4 = transpose_switch_frame(self.c2_4, self.cylinder_test.matrix)
#         v1, v2 = find_tangent_points_xy(P_cylinder_frame_4, S_cylinder_frame_4, self.r)
#         v1_v2_angle_xy = np.arccos(1.0-((v1[0]-v2[0])**2+(v1[1]-v2[1])**2)/(2*self.r**2))
#         print("v1_v2_angle_xy = ", v1_v2_angle_xy)
        
#         arctan2 = arctan2_corrected(v1, v2)
#         self.assertTrue(np.allclose(v1_v2_angle_xy, arctan2), 
#                         f"Test_4_compute_length_v1_v2_xy\nAngle must be equal to {v1_v2_angle_xy}, got: {arctan2}")
        
#         v1, v2 = find_tangent_points(P_cylinder_frame_4, S_cylinder_frame_4, self.r)
#         # ------
#         # Step 3
#         # ------
#         obstacle_tangent_point_inactive = determine_if_tangent_points_inactive_single_cylinder(v1,v2, self.r)
        
#         self.assertTrue(np.allclose(obstacle_tangent_point_inactive, False), 
#                         f"Test_2_compute_length_v1_v2_xy\nAngle must be equal to {False}, got: {obstacle_tangent_point_inactive}")
        
#         # ------
#         # Step 4
#         # ------
#         segment_length = segment_length_single_cylinder(obstacle_tangent_point_inactive, P_cylinder_frame_4, S_cylinder_frame_4, v1, v2, self.r)
#         print("segment length = ", segment_length)
        
# if __name__ == '__main__':
#     unittest.main()

    
    
    
    