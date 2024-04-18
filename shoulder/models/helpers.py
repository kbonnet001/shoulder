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
    def find_relaxed_poses(model: ModelAbstract, expand: bool = True) -> dict[str, np.array]:
        """
        Find the relaxed pose for each muscle that minimizes their respective muscle length

        Parameters
        ----------
        model: ModelAbstract
            The model to use
        muscle_index: int
            The muscle index to optimize
        expand: bool
            If the casadi functions should be expanded before the optimization

        Returns
        -------
            The relaxed pose for each muscle
        """
        # Declare some aliases
        n_q = model.n_q

        # Declare the decision variables
        q_mx = casadi.MX.sym("q", n_q, 1)
        x_mx = casadi.vertcat(q_mx)

        muscle_lengths_function = model.muscles_kinematics(q_mx)
        out = {}
        for i in range(model.n_muscles):
            # Declare the cost function
            f = casadi.Function("f", [q_mx], [muscle_lengths_function[i, 0]])
            if expand:
                f = f.expand()
            f = f(x_mx)

            # Declare the constraints
            g_mx = []
            lbg = np.array([])
            ubg = np.array([])

            g = casadi.Function("g", [x_mx], [casadi.vertcat(*g_mx)])
            if expand:
                g = g.expand()
            g = g(x_mx)

            # Declare the bounds of the optimization problem
            lbx = model.q_ranges[:, 0]
            ubx = model.q_ranges[:, 1]
            x0 = (lbx + ubx) / 2

            # Solve the optimization problem
            solver = casadi.nlpsol("solver", "ipopt", {"x": x_mx, "f": f, "g": g}, {"ipopt.print_level": 0})
            sol = solver(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)

            # Parse the results
            out[model.muscle_names[i]] = sol["x"]

        return out

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
        optimal_lengths = casadi.SX.sym("optimal_lengths", n_muscles, 1)
        emg_mx = casadi.MX.ones(n_muscles, 1)
        q_mx = casadi.MX.sym("q", n_q, 1)
        qdot_mx = casadi.MX.zeros(n_q, 1)

        # Compute the cost function jacobian
        for i in range(n_muscles):
            model.set_muscle_parameters(index=i, optimal_length=optimal_lengths_mx[i])

        flce = casadi.Function(
            "flce",
            [optimal_lengths_mx, q_mx],
            [model.muscle_force_coefficients(emg_mx, q_mx, qdot_mx)[1]],
            ["optimal_lengths", "q"],
            ["flce"],
        )
        if expand:
            flce = flce.expand()

        # Optimize for each muscle
        x = []
        for i in range(n_muscles):
            q = model.strongest_poses[model.muscle_names[i]]
            f = casadi.Function("cost", [optimal_lengths[i]], [-flce(optimal_lengths, q)[i, 0]])  # Maximize force
            if expand:
                f = f.expand()

            # Optimize for the current muscle
            x0 = float(model.muscles_kinematics(q, muscle_index=i))
            x.append(OptimizationHelpers.clipped_gradient_descent(f=f, x0=x0))

        # Set back the original optimal lengths
        for i in range(n_muscles):
            model.set_muscle_parameters(index=i, optimal_length=optimal_lengths_bak[i])

        return np.array(x).reshape(-1)

    @staticmethod
    def find_minimal_tendon_slack_lengths(
        model: ModelAbstract, emg: np.ndarray, q: np.array, qdot: np.array, expand: bool = True
    ) -> np.array:
        """
        Find values for the tendon slack lengths where the muscle starts to produce passive muscle forces
        """

        # Declare some aliases
        target = 0.002  # Target a passive force of almost 0.2% of the maximal force
        n_q = model.n_q
        n_muscles = model.n_muscles

        # Get the relaxed poses
        relaxed_poses = MuscleHelpers.find_relaxed_poses(model=model, expand=expand)

        # Save the original tendon slack lengths
        tendon_slack_lengths_bak = [
            model.get_muscle_parameter(i, MuscleParameter.TENDON_SLACK_LENGTH) for i in range(n_muscles)
        ]

        # Declare the decision and fixed variables
        tendon_slack_lengths_mx = casadi.MX.sym("tendon_slack_lengths", n_muscles, 1)
        emg_mx = casadi.MX.zeros(n_muscles, 1)
        q_mx = casadi.MX.sym("q", n_q, 1)

        for i in range(n_muscles):
            model.set_muscle_parameters(index=i, tendon_slack_length=tendon_slack_lengths_mx[i])

        flpe = casadi.Function(
            "flpe",
            [tendon_slack_lengths_mx, q_mx],
            [model.muscle_force_coefficients(emg_mx, q_mx)[0]],  # TODO use tendon force? if 0.00?
            ["tendon_slack_lengths", "q"],
            ["flpe"],
        )
        if expand:
            flpe = flpe.expand()

        x = np.ones(n_muscles) * 0.0001
        for i in range(n_muscles):
            # Evaluate the muscle force at relaxed pose
            q = relaxed_poses[model.muscle_names[i]]
            cost = casadi.Function(
                "force_at_q",
                [tendon_slack_lengths_mx],
                [flpe(tendon_slack_lengths=tendon_slack_lengths_mx, q=q)["flpe"][i] - target],
                ["tendon_slack_lengths"],
                ["cost"],
            )
            if expand:
                cost = cost.expand()

            x[i] = OptimizationHelpers.squeezing_optimization(lbx=0, ubx=1, cost_function=cost)
            if x[i] < 0.001:
                x[i] = 0.001

        # Set back the original tendon slack lengths
        for i in range(n_muscles):
            model.set_muscle_parameters(index=i, tendon_slack_length=tendon_slack_lengths_bak[i])

        return x
