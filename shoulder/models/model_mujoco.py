import mujoco
import numpy as np

from .enums import ControlsTypes, IntegrationMethods
from .helpers import parse_muscle_index
from .model_abstract import ModelAbstract


class ModelMujoco(ModelAbstract):
    def __init__(self, model_path: str):
        self._model_name = model_path.split("/")[-1].split(".xml")[0]
        self._model = mujoco.MjModel.from_xml_path(model_path)

    @property
    def name(self) -> str:
        return self._model_name

    @property
    def n_q(self) -> int:
        return self._model.nq

    @property
    def n_muscles(self) -> int:
        return self._model.na

    def muscles_kinematics(
        self, q: np.ndarray, qdot: np.ndarray = None, muscle_index: range | slice | int = None
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        muscle_index = parse_muscle_index(muscle_index, self.n_muscles)
        n_muscles = len(range(muscle_index.start, muscle_index.stop))

        data = mujoco.MjData(self._model)
        mujoco.mj_resetData(self._model, data)

        has_qdot = qdot is not None
        if qdot is None:
            qdot = np.zeros_like(q)

        length = np.ndarray((n_muscles, q.shape[1]))
        velocity = np.ndarray((n_muscles, q.shape[1]))
        for i, (q_tp, qdot_tp) in enumerate(zip(q.T, qdot.T)):
            mujoco.mj_resetData(self._model, data)
            data.qpos = q_tp
            data.qvel = qdot_tp
            mujoco.mj_step1(self._model, data)
            length[:, i] = data.actuator_length[muscle_index]
            velocity[:, i] = data.actuator_velocity[muscle_index]

        if has_qdot:
            return length, velocity
        else:
            return length

    def muscle_force_coefficients(
        self,
        emg: np.ndarray,
        q: np.ndarray,
        qdot: np.ndarray = None,
        muscle_index: int | range | slice | None = None,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise ValueError("muscle_force_coefficients is not compatible with Mujoco models")

    def muscle_force(
        self, emg: np.ndarray, q: np.ndarray, qdot: np.ndarray, muscle_index: int | range | slice | None = None
    ) -> np.ndarray:
        muscle_index = parse_muscle_index(muscle_index, self.n_muscles)
        n_muscles = len(range(muscle_index.start, muscle_index.stop))

        data = mujoco.MjData(self._model)

        force = np.ndarray((n_muscles, q.shape[1]))
        for i, (emg_tp, q_tp, qdot_tp) in enumerate(zip(emg.T, q.T, qdot.T)):
            mujoco.mj_resetData(self._model, data)
            data.qpos = q_tp
            data.qvel = qdot_tp
            data.ctrl = emg_tp
            mujoco.mj_forward(self._model, data)
            force[:, i] = np.abs(data.actuator_force[muscle_index])  # TODO: check why absolute

        return force

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
            data.ctrl = controls

            mujoco.mj_step(self._model, data)

            q.append(np.array(data.qpos))
            qdot.append(np.array(data.qvel))

        return np.array(q).T, np.array(qdot).T

    def animate(self, states: list[np.ndarray], allow_interaction: bool = False) -> None:
        import mujoco_viewer

        data = mujoco.MjData(self._model)
        mujoco.mj_resetData(self._model, data)
        viewer = mujoco_viewer.MujocoViewer(self._model, data)

        while True:
            for q, qdot in zip(states[0].T, states[1].T):
                mujoco.mj_resetData(self._model, data)
                data.qpos = q
                data.qvel = qdot

                if allow_interaction:
                    mujoco.mj_step(self._model, data)
                else:
                    mujoco.mj_step1(self._model, data)

                if not viewer.is_alive:
                    # Dynamics takes some time, so make sure to break if the viewer is closed
                    break
                viewer.render()

            if allow_interaction or not viewer.is_alive:
                # If we do allow user to interact with the model, we should not restart the animation
                break

        # If we allow interaction, then we can keep the animation running
        if allow_interaction:
            while viewer.is_alive:
                mujoco.mj_step(self._model, data)
                viewer.render()
