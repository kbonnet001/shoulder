import unittest
import numpy as np
import biorbd
from wrapping.step_1 import * 

class Step_1_test(unittest.TestCase):
    
    def setUp(self):
        model = biorbd.Model("models/Wu_DeGroote.bioMod")
        muscle_names = [model.muscle(i).name().to_string() for i in range(model.nbMuscles())]
        mus = model.muscle(muscle_names.index("PECM2"))
        
        # insertionPosition pour PECM2 (cf Wu_DeGroote.bioMod) : 0.016 -0.0354957 0.005
        self.point = np.array([0.016, -0.0354957, 0.005])
        q_initial = np.array([0.0, 0.0, 0.0, 0.0])
        
        segment_names = [model.segment(i).name().to_string() for i in range(model.nbSegment())]
        segment_index = segment_names.index("humerus_right")
        self.gcs_seg_0 = [gcs.to_array() for gcs in model.allGlobalJCS(q_initial)][segment_index]
        
        model.UpdateKinematicsCustom(q_initial)
        self.insertion_muscle = mus.musclesPointsInGlobal(model, q_initial)[-1].to_array()

    def test_switch_frame(self):
        point_global_frame = switch_frame(self.point, self.gcs_seg_0)
        self.assertTrue(np.allclose(point_global_frame, self.insertion_muscle), 
                        f"Test_switch_frame\nValue must be equal to {self.insertion_muscle}, got: {point_global_frame}")

    def test_transpose_switch_frame(self):
        point_local_frame = transpose_switch_frame(self.insertion_muscle, self.gcs_seg_0)
        self.assertTrue(np.allclose(point_local_frame, self.point), 
                        f"Test_transpose_switch_frame\nValue must be equal to {self.point}, got: {point_local_frame}")
        
if __name__ == '__main__':
    unittest.main()

    
    
    
    