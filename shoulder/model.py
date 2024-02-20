from typing import Callable

import biorbd
import numpy as np
from scipy import integrate


class Model:
    def __init__(self, model_path: str):
        self.model = biorbd.Model(model_path)

    @property
    def name(self):
        return self.model.path().absolutePath().to_string().split("/")[-1].split(".bioMod")[0]

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

    def muscles_kinematics(
        self, q: np.ndarray, qdot: np.ndarray = None, muscle_index: range | slice | int = None
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        muscle_index = _parse_muscle_index(muscle_index, self.n_muscles)

        n_muscles = len(range(muscle_index.start, muscle_index.stop))
        lengths = np.ndarray((n_muscles, q.shape[1]))
        velocities = np.ndarray((n_muscles, q.shape[1]))
        for i in range(q.shape[1]):
            if qdot is None:
                self.model.updateMuscles(q[:, i], True)
            else:
                self.model.updateMuscles(q[:, i], qdot[:, i], True)

            for j in range(self.model.nbMuscleTotal()):
                lengths[j, i] = self.model.muscle(j).length(self.model, q[:, i], False)
                if qdot is not None:
                    velocities[j, i] = self.model.muscle(j).velocity(self.model, q[:, i], qdot[:, i], False)

        if qdot is None:
            return lengths
        else:
            return lengths, velocities

    def muscle_force_coefficients(
        self,
        emg: np.ndarray,
        q: np.ndarray,
        qdot: np.ndarray = None,
        muscle_index: int | range | slice | None = None,
    ):
        muscle_index = _parse_muscle_index(muscle_index, self.n_muscles)
        if len(q.shape) == 1:
            q = q[:, np.newaxis]
        if qdot is not None and len(qdot.shape) == 1:
            qdot = qdot[:, np.newaxis]

        n_muscles = len(range(muscle_index.start, muscle_index.stop))

        out_flce = np.ndarray((n_muscles, q.shape[1]))
        out_fvce = np.ndarray((n_muscles, q.shape[1]))
        for i in range(q.shape[1]):
            if qdot is None:
                self.model.updateMuscles(q[:, i], True)
            else:
                self.model.updateMuscles(q[:, i], qdot[:, i], True)

            for j in range(muscle_index.start, muscle_index.stop):
                mus = _upcast_muscle(self.model.muscle(j))
                activation = biorbd.State(emg[j, i], emg[j, i])
                out_flce[j, i] = mus.FlCE(activation)
                if qdot is not None:
                    out_fvce[j, i] = mus.FvCE()

        if qdot is None:
            return out_flce
        else:
            return out_flce, out_fvce

    def muscle_force(
        self, emg: np.ndarray, q: np.ndarray, qdot: np.ndarray, muscle_index: int | range | slice | None = None
    ):
        muscle_index = _parse_muscle_index(muscle_index, self.n_muscles)
        if len(q.shape) == 1:
            q = q[:, np.newaxis]
        if len(qdot.shape) == 1:
            qdot = qdot[:, np.newaxis]

        n_muscles = len(range(muscle_index.start, muscle_index.stop))

        out_force = np.ndarray((n_muscles, q.shape[1]))
        for i in range(q.shape[1]):
            if qdot is None:
                self.model.updateMuscles(q[:, i], True)
            else:
                self.model.updateMuscles(q[:, i], qdot[:, i], True)

            for j in range(muscle_index.start, muscle_index.stop):
                mus = _upcast_muscle(self.model.muscle(j))
                activation = biorbd.State(emg[j, i], emg[j, i])
                out_force[j, i] = mus.force(activation)

        return out_force

    def forward_dynamics_muscles(self, t: float, x: np.ndarray, emg: np.ndarray):
        q = x[: self.n_q]
        qdot = x[self.n_q :]
        tau = self.model.muscularJointTorque(np.ones(emg.shape), q, qdot).to_array()
        return self.forward_dynamics(t, x, tau)

    def forward_dynamics(self, t: float, x: np.ndarray, tau: np.ndarray):
        q = x[: self.n_q]
        qdot = x[self.n_q :]
        qddot = self.model.ForwardDynamics(q, qdot, tau).to_array() * 0.0001
        return np.concatenate((qdot, qddot))


def _upcast_muscle(muscle: biorbd.Muscle) -> biorbd.HillType | biorbd.HillThelenType | biorbd.HillDeGrooteType:
    muscle_type_id = muscle.type()
    if muscle_type_id == biorbd.IDEALIZED_ACTUATOR:
        return biorbd.IdealizedActuator(muscle)
    elif muscle_type_id == biorbd.HILL:
        return biorbd.HillType(muscle)
    elif muscle_type_id == biorbd.HILL_THELEN:
        return biorbd.HillThelenType(muscle)
    elif muscle_type_id == biorbd.HILL_DE_GROOTE:
        return biorbd.HillDeGrooteType(muscle)
    else:
        raise ValueError(f"Muscle type {muscle_type_id} not supported")


def _parse_muscle_index(muscle_index: range | slice | int | None, n_muscles: int) -> slice:
    if muscle_index is None:
        return slice(0, n_muscles)
    elif isinstance(muscle_index, int):
        return slice(muscle_index, muscle_index + 1)
    elif isinstance(muscle_index, range):
        return slice(muscle_index.start, muscle_index.stop)
    else:
        raise ValueError("muscle_index must be an int, a range or a slice")
