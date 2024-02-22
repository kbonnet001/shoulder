import mujoco
import numpy as np

from .enums import ControlsTypes, IntegrationMethods
from .model_abstract import ModelAbstract


class ModelMujoco(ModelAbstract):
    def __init__(self, model_path: str):
        self._model = mujoco.MjModel.from_xml_path(model_path)

    @property
    def name(self) -> str:
        raise NotImplementedError("ModelMujoco.name")

    @property
    def n_q(self) -> int:
        return self._model.nq

    @property
    def n_muscles(self) -> int:
        return self._model.na

    def muscles_kinematics(
        self, q: np.ndarray, qdot: np.ndarray = None, muscle_index: range | slice | int = None
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError("ModelMujoco.muscles_kinematics")

    def muscle_force_coefficients(
        self,
        emg: np.ndarray,
        q: np.ndarray,
        qdot: np.ndarray = None,
        muscle_index: int | range | slice | None = None,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError("ModelMujoco.muscle_force_coefficients")

    def muscle_force(
        self, emg: np.ndarray, q: np.ndarray, qdot: np.ndarray, muscle_index: int | range | slice | None = None
    ) -> np.ndarray:
        raise NotImplementedError("ModelMujoco.muscle_force")

    def integrate(
        self,
        t: np.ndarray,
        states: np.ndarray,
        controls: np.ndarray,
        controls_type: ControlsTypes = ControlsTypes.TORQUE,
        integration_method: IntegrationMethods = IntegrationMethods.RK4,
    ) -> tuple[np.ndarray, np.ndarray]:

        if controls_type == ControlsTypes.EMG:
            # TODO: Implement EMG control
            pass
        elif controls_type == ControlsTypes.TORQUE:
            raise NotImplementedError("ModelMujoco.integrate TORQUE")
        else:
            raise NotImplementedError(f"Control {controls_type} not implemented")

        if integration_method == IntegrationMethods.RK45:
            raise ValueError("ModelMujoco.integrate does not support RK45")
        elif integration_method == IntegrationMethods.RK4:
            pass
        else:
            raise NotImplementedError(f"Integration method {integration_method} not implemented")

        data = mujoco.MjData(self._model)
        mujoco.mj_resetData(self._model, data)

        q = []
        qdot = []
        self._model.opt.timestep = t[1] - t[0]
        while data.time < t[-1] - t[0]:
            mujoco.mj_step(self._model, data)
            q.append(data.qpos)
            qdot.append(data.qvel)

        return np.array(q).T, np.array(qdot).T
