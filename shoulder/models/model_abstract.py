from abc import ABC, abstractmethod

from .enums import ControlsTypes, IntegrationMethods, MuscleParameter
from ..helpers import Vector, Scalar


class ModelAbstract(ABC):
    @property
    @property
    def name(self) -> str:
        """
        The name of the model
        """

    @property
    @abstractmethod
    def n_q(self) -> int:
        """
        The number of generalized coordinates
        """

    @property
    @abstractmethod
    def n_muscles(self) -> int:
        """
        The number of muscles
        """

    @property
    def muscle_names(self) -> list[str]:
        """
        The muscle names
        """

    @property
    @abstractmethod
    def relaxed_poses(self) -> map:
        """
        Get the relaxed poses (the pose where the muscle is expected to start producing passive force) for each muscle

        Returns
        -------
        map[str, np.ndarray]
            The relaxed poses for each muscle, the key is the muscle name and the value is the joint angles vector
        """

    @property
    @abstractmethod
    def strongest_poses(self) -> map:
        """
        Get the strongest poses for each muscle

        Returns
        -------
        map[str, np.ndarray]
            The strongest poses for each muscle, the key is the muscle name and the value is the joint angles vector
        """

    @abstractmethod
    def muscles_kinematics(
        self, q: Vector, qdot: Vector = None, muscle_index: range | slice | int = None
    ) -> Vector | tuple[Vector, Vector]:
        """
        Compute the muscle kinematics, that is the muscle length and velocity. If qdot is None, only the muscle length
        is computed. Otherwise, both the muscle length and velocity are returned.
        """

    @abstractmethod
    def muscle_force_coefficients(
        self,
        emg: Vector,
        q: Vector,
        qdot: Vector = None,
        muscle_index: int | range | slice | None = None,
    ) -> Vector | tuple[Vector, Vector, Vector]:
        """
        Compute the muscle force-length and force-velocity properties. If qdot is None, only the force-length property
        is computed (that is passive and contractive force-length properties). Otherwise, both the force-length and
        force-velocity properties are returned.
        """

    @abstractmethod
    def muscle_force(
        self, emg: Vector, q: Vector, qdot: Vector, muscle_index: int | range | slice | None = None
    ) -> Vector:
        """
        Compute the muscle forces
        """

    @abstractmethod
    def set_muscle_parameters(
        self, index: int, optimal_length: Scalar = None, tendon_slack_length: Scalar = None
    ) -> None:
        """
        Set the muscle parameters

        Parameters
        ----------
        index: int
            The muscle index
        optimal_length: Scalar
            The optimal length
        tendon_slack_length: Scalar
            The tendon slack length
        """

    @abstractmethod
    def get_muscle_parameter(self, index: int, parameter_to_get: MuscleParameter) -> Scalar:
        """
        Get the muscle parameters

        Parameters
        ----------
        index: int
            The muscle index
        parameter_to_get: MuscleParameter
            The parameter to get

        Returns
        -------
        Scalar
            The muscle parameter requested
        """

    @abstractmethod
    def integrate(
        self,
        t_span: tuple[float, float],
        states: Vector,
        controls: Vector,
        t_eval: Vector = None,
        controls_type: ControlsTypes = ControlsTypes.TORQUE,
        integration_method: IntegrationMethods = IntegrationMethods.RK45,
    ) -> tuple[Vector, Vector]:
        """
        Integrate the model
        """

    @abstractmethod
    def forward_dynamics(
        self,
        q: Vector,
        qdot: Vector,
        controls: Vector,
        controls_type: ControlsTypes = ControlsTypes.TORQUE,
    ) -> tuple[Vector, Vector]:
        """
        Integrate the model forward dynamics
        """

    @abstractmethod
    def animate(self, states: list[Vector], *args, **kwargs) -> None:
        """
        Animate the model
        """
