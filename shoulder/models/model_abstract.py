from abc import ABC, abstractproperty, abstractmethod

from .enums import ControlsTypes, IntegrationMethods
from .helpers import Vector, Scalar


class ModelAbstract(ABC):
    @abstractproperty
    def name(self) -> str:
        """
        The name of the model
        """

    @abstractproperty
    def n_q(self) -> int:
        """
        The number of generalized coordinates
        """

    @abstractproperty
    def n_muscles(self) -> int:
        """
        The number of muscles
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
    def set_muscle_parameters(self, index: int, optimal_length: Scalar) -> None:
        """
        Set the muscle parameters

        Parameters
        ----------
        index: int
            The muscle index
        optimal_length: Scalar
            The optimal length
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
