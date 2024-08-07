import numpy as np
import biorbd
from neural_networks.file_directory_operations import *

def compute_muscle_force_origin_insertion_nul(muscle_index, lmt, model_one_muscle = biorbd.Model("models/oneMuscle.bioMod")) :
    """
    Compute muscle force with lmt
    This is a temporarily function !
    
    We suppose that origin point = 0, 0, 0 and insertion point = 0, 0, 0
    Then, insertion point = 0, 0, lmt
    Please, paid attention to the file 'oneMuscle.bioMod'
    For the moment, there is only PECM2 and PECM3 with modified origin and insertion points
    
    """
    q = np.array([0])
    qdot = np.array([0])
    
    mus = model_one_muscle.muscle(muscle_index) 
    mus.position().setInsertionInLocal(np.array([0, 0, lmt])) 
    
    states = model_one_muscle.stateSet()
    for state in states:
        state.setActivation(1) # 1 ==> 100% activation
    f = model_one_muscle.muscleForces(states, q, qdot).to_array()

    print(f"f: {f[muscle_index]}")
    if f[muscle_index] >= 5000 : 
        print("ERROR : force >= 5000 !!!")
    
    return f[muscle_index]
    

def compute_torque(dlmt_dq, f) : 
    print("dlmt_dq = ", dlmt_dq)
    print("f = ", f)
    # torque = []
    # for i in range(len(f)) : 
    #     torque.append(sum(np.dot(- np.transpose(dlmt_dq[i]), f[i])))
    return sum(np.dot(- np.transpose(dlmt_dq), f))

def compute_torque_from_lmt_and_dlmt_dq(muscle_index, lmt, dlmt_dq) : 
    model_one_muscle = biorbd.Model("models/oneMuscle.bioMod")
    f = compute_muscle_force_origin_insertion_nul(muscle_index, lmt, model_one_muscle)
    return compute_torque(dlmt_dq, f)