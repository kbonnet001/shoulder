import biorbd
import numpy as np

from .model_abstract import ModelAbstract
from .helpers import parse_muscle_index


class ModelBiorbd(ModelAbstract):
    def __init__(self, model_path: str):
        self._model = biorbd.Model(model_path)

    @property
    def biorbd_model(self) -> biorbd.Model:
        return self._model

    @property
    def name(self) -> str:
        return self._model.path().absolutePath().to_string().split("/")[-1].split(".bioMod")[0]

    @property
    def n_q(self) -> int:
        return self._model.nbQ()

    @property
    def n_muscles(self) -> int:
        return self._model.nbMuscleTotal()

    def muscles_kinematics(
        self, q: np.ndarray, qdot: np.ndarray = None, muscle_index: range | slice | int = None
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        muscle_index = parse_muscle_index(muscle_index, self.n_muscles)

        n_muscles = len(range(muscle_index.start, muscle_index.stop))
        lengths = np.ndarray((n_muscles, q.shape[1]))
        velocities = np.ndarray((n_muscles, q.shape[1]))
        for i in range(q.shape[1]):
            if qdot is None:
                self._model.updateMuscles(q[:, i], True)
            else:
                self._model.updateMuscles(q[:, i], qdot[:, i], True)

            for j in range(self._model.nbMuscleTotal()):
                lengths[j, i] = self._model.muscle(j).length(self._model, q[:, i], False)
                if qdot is not None:
                    velocities[j, i] = self._model.muscle(j).velocity(self._model, q[:, i], qdot[:, i], False)

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
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
        muscle_index = parse_muscle_index(muscle_index, self.n_muscles)
        if len(q.shape) == 1:
            q = q[:, np.newaxis]
        if qdot is not None and len(qdot.shape) == 1:
            qdot = qdot[:, np.newaxis]

        n_muscles = len(range(muscle_index.start, muscle_index.stop))

        out_flpe = np.ndarray((n_muscles, q.shape[1]))
        out_flce = np.ndarray((n_muscles, q.shape[1]))
        out_fvce = np.ndarray((n_muscles, q.shape[1]))
        for i in range(q.shape[1]):
            if qdot is None:
                self._model.updateMuscles(q[:, i], True)
            else:
                self._model.updateMuscles(q[:, i], qdot[:, i], True)

            for j in range(muscle_index.start, muscle_index.stop):
                mus = _upcast_muscle(self._model.muscle(j))
                activation = biorbd.State(emg[j, i], emg[j, i])
                out_flpe[j, i] = mus.FlPE()
                out_flce[j, i] = mus.FlCE(activation)
                if qdot is not None:
                    out_fvce[j, i] = mus.FvCE()

        if qdot is None:
            return out_flpe, out_flce
        else:
            return out_flpe, out_flce, out_fvce

    def muscle_force(
        self, emg: np.ndarray, q: np.ndarray, qdot: np.ndarray, muscle_index: int | range | slice | None = None
    ) -> np.ndarray:
        muscle_index = parse_muscle_index(muscle_index, self.n_muscles)
        if len(q.shape) == 1:
            q = q[:, np.newaxis]
        if len(qdot.shape) == 1:
            qdot = qdot[:, np.newaxis]

        n_muscles = len(range(muscle_index.start, muscle_index.stop))

        out_force = np.ndarray((n_muscles, q.shape[1]))
        for i in range(q.shape[1]):
            if qdot is None:
                self._model.updateMuscles(q[:, i], True)
            else:
                self._model.updateMuscles(q[:, i], qdot[:, i], True)

            for j in range(muscle_index.start, muscle_index.stop):
                mus = _upcast_muscle(self._model.muscle(j))
                activation = biorbd.State(emg[j, i], emg[j, i])
                out_force[j, i] = mus.force(activation)

        return out_force

    def forward_dynamics_muscles(self, t: float, x: np.ndarray, emg: np.ndarray) -> np.ndarray:
        q = x[: self.n_q]
        qdot = x[self.n_q :]
        states = self._model.stateSet()
        for k in range(self._model.nbMuscles()):
            states[k].setActivation(emg[k])
        tau = self._model.muscularJointTorque(states, q, qdot).to_array()
        return self.forward_dynamics(t, x, tau)

    def forward_dynamics(self, t: float, x: np.ndarray, tau: np.ndarray) -> np.ndarray:
        q = x[: self.n_q]
        qdot = x[self.n_q :]
        qddot = self._model.ForwardDynamics(q, qdot, tau).to_array() * 0.0001
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
