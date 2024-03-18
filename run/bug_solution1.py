import biorbd_casadi
import casadi


class ModelBiorbd:
    def __init__(self, path: str):
        self._path = path
        self._model = biorbd_casadi.Model(path)

        self._q = None
        self._qdot = None
        self._tau = None
        self._forward_dynamics = None

    @property
    def nb_q(self):
        return self._model.nbQ()

    def set_states(self, q: casadi.MX, qdot: casadi.MX):
        self._q = q
        self._qdot = qdot

    def set_controls(self, tau: casadi.MX):
        self._tau = tau

    @property
    def forward_dynamics(self):
        # This saves the forward dynamics to avoid recomputing it, which prevents from internally adding dummy variables
        # This has the side effect that if the model is modified, the forward dynamics won't reflect the changes.
        # Unless we provide a "reset" method, which introduces a new problem: the user must remember (and know when) to call it.
        if self._forward_dynamics is None:
            if self._q is None or self._qdot is None or self._tau is None:
                raise RuntimeError("You must set the states and controls before calling forward_dynamics")
            self._forward_dynamics = self._model.ForwardDynamics(self._q, self._qdot, self._tau).to_mx()
        return self._forward_dynamics


# Déclaration du modèle
cx = casadi.SX
model = ModelBiorbd("models/Wu_DeGroote.bioMod")
n_q = model.nb_q

# Premier appel de la ForwardDynamics
q_mx = casadi.MX.sym("q", n_q, 1)
qdot_mx = casadi.MX.sym("qdot", n_q, 1)
tau_mx = casadi.MX.sym("tau", n_q, 1)
model.set_states(q_mx, qdot_mx)
model.set_controls(tau_mx)

fd_func = casadi.Function(
    "fd", [q_mx, qdot_mx, tau_mx], [model.forward_dynamics], ["q", "qdot", "tau"], ["qdot_out"]
).expand()
fd_func2 = casadi.Function(
    "fd", [q_mx, qdot_mx, tau_mx], [model.forward_dynamics], ["q", "qdot", "tau"], ["qdot_out"]
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
