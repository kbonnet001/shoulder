from abc import ABC, abstractproperty, abstractmethod

import numpy as np


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
        self, q: np.ndarray, qdot: np.ndarray = None, muscle_index: range | slice | int = None
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        Compute the muscle kinematics, that is the muscle length and velocity. If qdot is None, only the muscle length
        is computed. Otherwise, both the muscle length and velocity are returned.
        """

    def muscle_force_coefficients(
        self,
        emg: np.ndarray,
        q: np.ndarray,
        qdot: np.ndarray = None,
        muscle_index: int | range | slice | None = None,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the muscle force-length and force-velocity properties. If qdot is None, only the force-length property
        is computed (that is passive and contractive force-length properties). Otherwise, both the force-length and
        force-velocity properties are returned.
        """

    def muscle_force(
        self, emg: np.ndarray, q: np.ndarray, qdot: np.ndarray, muscle_index: int | range | slice | None = None
    ) -> np.ndarray:
        """
        Compute the muscle forces
        """

    def forward_dynamics_muscles(self, t: float, x: np.ndarray, emg: np.ndarray) -> np.ndarray:
        """
        Compute the forward dynamics of the model with the muscle activations as controls
        """

    def forward_dynamics(self, t: float, x: np.ndarray, tau: np.ndarray) -> np.ndarray:
        """
        Compute the forward dynamics of the model with the joint torques as controls
        """
