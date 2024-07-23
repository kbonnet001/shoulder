import numpy as np
import biorbd

def test_muscle_force() : 
    m = biorbd.Model("models/oneMuscle.bioMod")
    m2 = biorbd.Model("models/Wu_DeGroote.bioMod")
    
    
    q = np.array([0])
    qdot = np.array([0])


    m.muscle(0).position().setInsertionInLocal(np.array([0, 0, 0.20]))
    m.muscle(1).position().setInsertionInLocal(np.array([0, 0, 0.21]))

    states = m.stateSet()
    for state in states:
        state.setActivation(1)
    f = m.muscleForces(states, q, qdot).to_array()

    print(f"f: {f}")
