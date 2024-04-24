from abc import ABC, abstractmethod

import numpy as np

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
    def q_ranges(self) -> np.ndarray:
        """
        The joint ranges in n_q x 2 matrix (min, max)
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

    @abstractmethod
    def optimize_muscle_parameters(
        self, use_predefined_muscle_ratio_values: bool = True, robust_optimization: bool = False, expand: bool = True
    ) -> None:
        """
        Interface for calling the "optimize_muscle_parameters" and having the results dispatched to the current model

        Parameters
        ----------
        use_predefined_muscle_ratio_values: bool
            If the predefined muscle ratio of the model values should be used
        robust_optimization: bool
            If the optimization should be robust
        expand: bool
            If the optimization should expand the casadi Functions (faster but takes more RAM)

        Returns
        -------
        Changes the muscle parameters of the model, returns nothing
        """

    @property
    @abstractmethod
    def relaxed_pose(self) -> np.ndarray:
        """
        The relaxed pose
        """

    @abstractmethod
    def ranged_relaxed_poses(self, limit: float, n_elements: int) -> np.ndarray:
        """
        All te poses that are within the relaxed pose ranges. Rows are the generalized coordinates, cols are each pose.
        The more a pose appears in the list, the more weight it has in the optimization problem. The first is the relaxed
        pose itself

        Parameters
        ----------
        limit: float
            The limit to push the relaxed pose to each side
        n_elements: int
            The number of elements to generate

        Returns
        -------
        np.ndarray
            The relaxed poses, the rows are the generalized coordinates, the columns are the poses
        """

    @property
    @abstractmethod
    def strongest_poses(self) -> dict[str, np.ndarray]:
        """
        Get the strongest poses for each muscle

        Returns
        -------
        dict[str, np.ndarray]
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
