import numpy as np
import biorbd
from neural_networks.functions_data_generation import update_points_position

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


# def compute_torque(dlmt_dq, f):
#     # Assurez-vous que dlmt_dq est un tableau numpy
#     dlmt_dq = np.array(dlmt_dq)
    
#     # Vérifiez que f est bien un scalaire
#     if isinstance(f, (int, float)):
#         # Calculez le torque en utilisant le produit scalaire
#         torque = np.dot(-dlmt_dq, f)
        
#         # Affichez les valeurs pour le débogage
#         print("dlmt_dq = ", dlmt_dq)
#         print("f = ", f)
#         print("np.dot(-dlmt_dq, f) = ", torque)
        
#         return torque
#     else:
#         raise ValueError("f doit être un scalaire (int ou float).")

    
model_one_muscle = biorbd.Model("models/oneMuscle.bioMod")
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



# faire une fonction pour calculer le torque et ajouter au sheet de donnee

# fini : 
    # revoir le fichier one muscle et verifier aue c'est bien PECM2 et PECM3 avec juste origin et insertion
    # faire une fonction propre pour avoir F et avec un warning pour dire aue C,est possible aue pour PECM2 et PECM3 