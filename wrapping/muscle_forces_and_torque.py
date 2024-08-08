import numpy as np
import biorbd
from neural_networks.file_directory_operations import *

def compute_muscle_force_origin_insertion_nul(muscle_index, q, qdot, alpha, lmt, model_one_muscle = biorbd.Model("models/oneMuscle.bioMod")) :
    """
    Compute muscle force with lmt
    This is a temporarily function !
    
    We suppose that origin point = 0, 0, 0 and insertion point = 0, 0, 0
    Then, insertion point = 0, 0, lmt
    Please, paid attention to the file 'oneMuscle.bioMod'
    For the moment, there is only PECM2 and PECM3 with modified origin and insertion points
    
    """
    # q = np.array([q]) # doit etre de taille 1
    q = np.array([0])
    qdot = np.array([0])
    
    mus = model_one_muscle.muscle(muscle_index) #
    mus.position().setInsertionInLocal(np.array([0, 0, lmt])) #
    # print(mus.musclesPointsInGlobal(model_one_muscle, q)[0].to_array())
    # print(mus.musclesPointsInGlobal(model_one_muscle, q)[-1].to_array())
    
    states = model_one_muscle.stateSet()
    for state in states:
        state.setActivation(alpha) # example 1 ==> 100% activation
    f = model_one_muscle.muscleForces(states, q, qdot).to_array()

    print(f"f: {f[muscle_index]}")
    if f[muscle_index] >= 5000 : 
        print("ERROR : force >= 5000 !!!")
    
    return f[muscle_index] 
    
def compute_fm_nul(muscle_index, q, qdot, alpha, lmt) : 
    fm = []
    for q_i, qdot_i, alpha_i in zip(q, qdot, alpha) :
        fm.append(compute_muscle_force_origin_insertion_nul(muscle_index, q_i, qdot_i, alpha_i, lmt))
    return np.array(fm)

def compute_fm_muscle_index(model_biorbd, muscle_index, q, qdot, alpha) : 
    f = compute_fm(model_biorbd, q, qdot, alpha)

    print(f"f: {f[muscle_index]}")
    if f[muscle_index] >= 5000 : 
        print("ERROR : force >= 5000 !!!")
    
    return f[muscle_index] 



def comupte_torque(model_biorbd, f) : 
    return model_biorbd.muscularJointTorque(f).to_array()

################################################################
def compute_fm(model_biorbd, q, qdot, alpha) : 
    states = model_biorbd.stateSet()
    for state in states:
        state.setActivation(alpha) # example 1 ==> 100% activation
    f = model_biorbd.muscleForces(states, q, qdot).to_array()
    
    return f

def compute_torque(dlmt_dq, f, limit = 5000) : 
    print("dlmt_dq = ", dlmt_dq)
    print("f = ", f)
    
    f_sup_limit = False
    # Security
    if f >= limit : 
        f_sup_limit = True

    # torque = []
    # for i in range(len(f)) : 
    #     torque.append(sum(np.dot(- np.transpose(dlmt_dq[i]), f[i])))
    return sum(np.dot(- np.transpose(dlmt_dq), f)), f_sup_limit

def get_fm_and_torque(model_biorbd, muscle_index, q, qdot, alpha) : 
    f_sup_limit = False
    fm = compute_fm(model_biorbd, q, qdot, alpha) 
    tau = model_biorbd.muscularJointTorque(fm).to_array()
    if fm[muscle_index] >= 5000 : 
        print("ERROR : force >= 5000 !!!")
        f_sup_limit = True
    
    return fm[muscle_index], tau[muscle_index], f_sup_limit
#############################################################

def compute_torque_from_lmt_and_dlmt_dq(muscle_index, q, qdot, alpha, lmt, dlmt_dq) : 
    model_one_muscle = biorbd.Model("models/oneMuscle.bioMod")
    f = compute_muscle_force_origin_insertion_nul(muscle_index, q, qdot, alpha, lmt, model_one_muscle)
    return compute_torque(dlmt_dq, f)