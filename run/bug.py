import biorbd_casadi
import casadi
import numpy as np


# Déclaration du modèle
cx = casadi.SX
model = biorbd_casadi.Model("models/Wu_DeGroote.bioMod")
n_q = model.nbQ()

# Premier appel de la ForwardDynamics
q_mx = casadi.MX.sym("q", n_q, 1)
qdot_mx = casadi.MX.sym("qdot", n_q, 1)
tau_mx = casadi.MX.sym("tau", n_q, 1)
# model.setGravity(casadi.MX([0, 10, 0]))


new_jcs = biorbd_casadi.RotoTrans.fromEulerAngles(
    rot=np.array([0, 0, 0]), trans=np.array([1000, 1000, 1000]), seq="xyz"
)
model.segment(6).setLocalJCS(model, new_jcs)
fd = model.ForwardDynamics(q_mx, qdot_mx, tau_mx).to_mx()
fd_func = casadi.Function("fd", [q_mx, qdot_mx, tau_mx], [fd], ["q", "qdot", "tau"], ["qdot_out"]).expand()

# Deuxième appel de la ForwardDynamics, des variables se sont ajoutées dans le modèle
q2_mx = casadi.MX.sym("q", n_q, 1)
qdot2_mx = casadi.MX.sym("qdot", n_q, 1)
tau2_mx = casadi.MX.sym("tau", n_q, 1)
fd2 = model.ForwardDynamics(q2_mx, qdot2_mx, tau2_mx).to_mx()
# La ligne suivante cause des free variable, car le premier appel à ForwardDynamics a "ajouté" des variables DANS le modèle
fd_func2 = casadi.Function("fd", [q2_mx, qdot2_mx, tau2_mx], [fd2], ["q", "qdot", "tau"], ["qdot_out"]).expand()
# Il faut donc ajouter les variables du premier appel en "dummy" pour que le graphe MX se monte correctement
fd_func2 = casadi.Function(
    "fd",
    [q2_mx, qdot2_mx, tau2_mx],
    [fd2],
    ["q", "qdot", "tau"],
    ["qdot_out"],
).expand()

# Si on compare ici fd et fd2, on les graph MX sont différents, et plus complexe pour le deuxième appel
# print(fd)
# print(fd2)

# Si on effondre le graphe en SX
q_cx = cx.sym("q", n_q, 1)
qdot_cx = cx.sym("qdot", n_q, 1)
tau_cx = cx.sym("tau", n_q, 1)
qdot = fd_func(q=q_cx, qdot=qdot_cx, tau=tau_cx)["qdot_out"]
qdot2 = fd_func2(q=q_cx, qdot=qdot_cx, tau=tau_cx)["qdot_out"]

# Là on obtient effectivement le même graph, prouvant que les variables supplémentaires étaient effectivement dummy
print(qdot)
print(qdot2)


# Et on peut s'en convaincre en calculant les jacobiennes
jacobian = casadi.jacobian(qdot, qdot_cx)
jacobian2 = casadi.jacobian(qdot2, qdot_cx)
print(jacobian)
print(jacobian2)
