from typing import Callable

import biorbd
import numpy as np
from scipy import integrate


class Model:
    def __init__(self, model_path: str):
        self.model = biorbd.Model(model_path)

    @property
    def n_q(self):
        return self.model.nbQ()

    @property
    def n_muscles(self):
        return self.model.nbMuscleTotal()

    def center_of_mass(self, q: np.ndarray):
        return np.array([self.model.CoM(tp).to_array() for tp in q.T])

    def center_of_mass_velocity(self, q: np.ndarray, qdot: np.ndarray):
        return np.array([self.model.CoMdot(tp, qdot[:, i]).to_array() for i, tp in enumerate(q.T)])

    def forward_dynamics_muscles(self, t: float, x: np.ndarray, emg: np.ndarray):
        q = x[: self.n_q]
        qdot = x[self.n_q :]
        tau = self.model.muscularJointTorque(emg, q, qdot).to_array()
        return self.forward_dynamics(t, x, tau)

    def forward_dynamics(self, t: float, x: np.ndarray, tau: np.ndarray):
        q = x[: self.n_q]
        qdot = x[self.n_q :]
        qddot = self.model.ForwardDynamics(q, qdot, tau).to_array() * 0.0001
        return np.concatenate((qdot, qddot))
