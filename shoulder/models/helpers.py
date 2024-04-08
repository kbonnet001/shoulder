import numpy as np
import casadi

from .model_abstract import ModelAbstract
from .enums import MuscleParameter
from ..helpers import OptimizationHelpers


class MuscleHelpers:
    @staticmethod
    def parse_muscle_index(muscle_index: range | slice | int | None, n_muscles: int) -> slice:
        if muscle_index is None:
            return slice(0, n_muscles)
        elif isinstance(muscle_index, int):
            return slice(muscle_index, muscle_index + 1)
        elif isinstance(muscle_index, range):
            return slice(muscle_index.start, muscle_index.stop)
        else:
            raise ValueError("muscle_index must be an int, a range or a slice")

    @staticmethod
    def find_optimal_length_assuming_strongest_pose(model: ModelAbstract, expand: bool = True) -> np.array:
        """
        Find values for the optimal muscle lengths where each muscle produces maximal force at their respective strongest
        pose. We could use IPOPT for that, but since the initial guess is so poor, sometimes we get to 0N force, which
        confuses the optimizer. This function uses a custom gradient descent algorithm that is more robust to this
        kind of problem.

        The method resets the model to its original state after the optimization

        Parameters
        ----------
        model: ModelAbstract
            The model to use
        expand: bool
            If the casadi functions should be expanded before the optimization

        Returns
        -------
        np.array
            The optimized optimal muscle lengths
        """

        # Declare some aliases
        n_q = model.n_q
        n_muscles = model.n_muscles
        optimal_lengths_bak = [model.get_muscle_parameter(i, MuscleParameter.OPTIMAL_LENGTH) for i in range(n_muscles)]

        # Declare the decision and fixed variables
        optimal_lengths_mx = casadi.MX.sym("optimal_lengths", n_muscles, 1)
        emg_mx = casadi.MX.ones(n_muscles, 1)
        q_mx = casadi.MX.sym("q", n_q, 1)
        qdot_mx = casadi.MX.zeros(n_q, 1)

        # Compute the cost function jacobian
        for i in range(n_muscles):
            model.set_muscle_parameters(index=i, optimal_length=optimal_lengths_mx[i])
        muscle_forces_coefficients_jacobian = casadi.Function(
            "jacobian",
            [optimal_lengths_mx, q_mx],
            [casadi.jacobian(model.muscle_force_coefficients(emg_mx, q_mx, qdot_mx)[1], optimal_lengths_mx)],
            ["optimal_lengths", "q"],
            ["jacobian"],
        )
        if expand:
            muscle_forces_coefficients_jacobian = muscle_forces_coefficients_jacobian.expand()

        # Optimize for each muscle
        x = np.ndarray(n_muscles) * np.nan
        for i in range(n_muscles):
            # Evaluate the jacobian at q
            q = model.strongest_poses[model.muscle_names[i]]
            jaco = casadi.Function(
                "jacobian_at_q",
                [optimal_lengths_mx],
                [muscle_forces_coefficients_jacobian(optimal_lengths=optimal_lengths_mx, q=q)["jacobian"][i, i]],
                ["optimal_lengths"],
                ["jacobian_at_q"],
            )
            if expand:
                jaco = jaco.expand()

            # Optimize for the current muscle
            flce_initial_guess = model.muscles_kinematics(q)[i, 0] / 2
            x[i] = OptimizationHelpers.simple_gradient_descent(x0=flce_initial_guess, cost_function_jacobian=jaco)

        # Set back the original optimal lengths
        for i in range(n_muscles):
            model.set_muscle_parameters(index=i, optimal_length=optimal_lengths_bak[i])

        return x

    @staticmethod
    def find_minimal_tendon_slack_lengths(
        model: ModelAbstract, emg: np.ndarray, q: np.array, qdot: np.array, expand: bool = True
    ) -> np.array:
        """
        Find values for the tendon slack lengths where the muscle starts to produce passive muscle forces
        """

        # Declare some aliases
        target = 0.02
        n_q = model.n_q
        n_muscles = model.n_muscles
        tendon_slack_lengths_bak = [
            model.get_muscle_parameter(i, MuscleParameter.TENDON_SLACK_LENGTH) for i in range(n_muscles)
        ]

        # Declare the decision and fixed variables
        tendon_slack_lengths_mx = casadi.MX.sym("tendon_slack_lengths", n_muscles, 1)
        emg_mx = casadi.MX.zeros(n_muscles, 1)
        q_mx = casadi.MX.sym("q", n_q, 1)
        qdot_mx = casadi.MX.zeros(n_q, 1)

        for i in range(n_muscles):
            model.set_muscle_parameters(index=i, tendon_slack_length=tendon_slack_lengths_mx[i])

        muscle_forces = casadi.Function(
            "muscle_forces",
            [tendon_slack_lengths_mx, q_mx],
            [model.muscle_force(emg_mx, q_mx, qdot_mx)],
            ["tendon_slack_lengths", "q"],
            ["forces"],
        )
        if expand:
            muscle_forces = muscle_forces.expand()

        x = np.ones(n_muscles) * 0.0001
        for i in range(n_muscles):
            # Evaluate the muscle force at q
            q = model.relaxed_poses[model.muscle_names[i]]
            force = casadi.Function(
                "force_at_q",
                [tendon_slack_lengths_mx],
                [muscle_forces(tendon_slack_lengths=tendon_slack_lengths_mx, q=q)["forces"][i] - target],
                ["tendon_slack_lengths"],
                ["force_at_q"],
            )
            if expand:
                force = force.expand()
            x[i] = OptimizationHelpers.squeezing_optimization(lbx=0, ubx=1, cost_function=force)

        # Set back the original tendon slack lengths
        for i in range(n_muscles):
            model.set_muscle_parameters(index=i, tendon_slack_length=tendon_slack_lengths_bak[i])

        return x
