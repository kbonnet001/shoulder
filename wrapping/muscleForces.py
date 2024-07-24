import numpy as np
import biorbd

def test_muscle_force() : 
    m = biorbd.Model("models/oneMuscle.bioMod")
    m2 = biorbd.Model("models/Wu_DeGroote.bioMod")
    
    q = np.array([0])
    qdot = np.array([0])

    mus = m.muscle(0) 
    mus.position().setInsertionInLocal(np.array([0, 0, 0.20])) # 0.20 a changer par lmt
    mus.musclesPointsInGlobal(m, q)[0].to_array()
    m.muscle(1).position().setInsertionInLocal(np.array([0, 0, 0.21]))

    mus2 = m2.muscle(0) 
    mus2.position().setInsertionInLocal(np.array([0, 0, 0.20])) # 0.20 a changer par lmt
    mus2.musclesPointsInGlobal(m2, q)[0].to_array()

    states = m.stateSet()
    for state in states:
        state.setActivation(1)
    f = m.muscleForces(states, q, qdot).to_array()

    print(f"f: {f}")

# en gros on aimerait changer la position origin et insertion du muscle pour mettre a 000 et ainsi mettre aue le 
# z aui change avec le bon etirement = lmt
# le probleme cest aue cest pas sur auon puisse faire cela donc peut ere oblige d,utiliser modele un muscle 
# pour eviter de faire des betises