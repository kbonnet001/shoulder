from typing import Callable

import casadi
import numpy as np

type Vector = np.ndarray | casadi.SX | casadi.MX | casadi.DM
type Scalar = float | int | casadi.SX | casadi.MX | casadi.DM


class VectorHelpers:
    def concatenate(*args: Vector) -> Vector:
        if isinstance(args[0], (casadi.SX, casadi.MX, casadi.DM)):
            return casadi.vertcat(*args)
        elif isinstance(args[0], np.ndarray):
            return np.concatenate(args, axis=0)
        else:
            raise ValueError("Unsupported type for concatenation")

    def initialize(requested_type: type, n_rows: int, n_cols: int) -> Vector:
        if requested_type in (casadi.SX, casadi.MX, casadi.DM):
            return requested_type(n_rows, n_cols)
        elif requested_type == np.ndarray:
            return requested_type((n_rows, n_cols))
        else:
            raise ValueError("Unsupported type for initialization")


class OptimizationHelpers:
    @staticmethod
    def newton_descent(f: casadi.Function, x0: np.ndarray, tol=1e-8, max_iter=100):
        """
        Newton descent algorithm to find the minimum of a function

        Parameters
        ----------
        f: casadi.Function
            The function to minimize
        x0: np.ndarray
            The initial guess
        tol: float
            The tolerance to consider the algorithm converged
        max_iter: int
            The maximum number of iterations

        Returns
        -------
        The optimized value
        """
        # Check the initial guess
        if len(x0.shape) < 1:
            x0 = x0.reshape((-1))
        elif len(x0.shape) >= 2:
            raise ValueError("The initial guess must be a 1D array")
        x = x0

        # Define the decision variable
        x_sx = casadi.SX.sym("x", x.shape[0], 1)

        # Define the gradient and hessian of the objective function
        evaluate = casadi.Function("evaluate", [x_sx], casadi.hessian(f(x_sx), x_sx), ["x"], ["hessian", "gradient"])
        for _ in range(max_iter):
            # Evaluate the gradient and hessian at the current point
            tp = evaluate(x=x)

            # Perform the Newton update step
            delta_x = casadi.solve(tp["hessian"], -tp["gradient"])
            x += delta_x

            # Check convergence
            if casadi.norm_inf(delta_x) < tol:
                break

        return x

    @staticmethod
    def clipped_gradient_descent(f: casadi.Function, x0: float, tol: float = 1e-8):
        """
        Simple gradient descent algorithm, if the algorithm is stuck bouncing between two values, we increase the bouncing
        modifier which reduces the step size. This works makes the assumption that the function only has one local
        minimum which is the global minimum. This only works for scalar functions.

        Parameters
        ----------
        f: casadi.Function
            The function to minimize
        x0: float
            The initial guess
        tol: float
            The tolerance to consider the algorithm converged

        Returns
        -------
        The optimized value
        """

        if x0 == 0 or not isinstance(x0, float):
            raise ValueError("The initial guess must be a float and not 0")

        # Define the decision variable
        x_sx = casadi.SX.sym("x", 1, 1)
        jacobian = casadi.Function("jacobian", [x_sx], [casadi.jacobian(f(x_sx), x_sx)], ["x"], ["jacobian"])

        previous_sign = 1
        nb_sign_bouncing = 0
        x = x0
        while True:
            # Get the muscle force
            jaco_value = -jacobian(x)

            # If the change is smaller than the threshold, we are at an optimal value
            if np.sum(np.abs(jaco_value)) < tol:
                break

            # This bouncing mechanism is used to avoid the optimizer to get stuck bouncing between two values around
            # the optimal value
            if previous_sign * jaco_value < 0:
                nb_sign_bouncing += 1
            previous_sign = np.sign(jaco_value)
            bouncing_modifier = 10 ** (-nb_sign_bouncing)

            x += x * np.clip(jaco_value, -0.1 * bouncing_modifier, 0.1 * bouncing_modifier)

        return x

    @staticmethod
    def squeezing_optimization(lbx: float, ubx: float, cost_function: Callable, threshold: float = 1e-8):
        """
        Squeezing optimization algorithm, this algorithm is used to find the optimal value of a that only has a single
        optimal point, but is not diffentiable.

        Parameters
        ----------
        lbx: float
            The lower bound of the optimization
        ubx: float
            The upper bound of the optimization
        cost_function: Callable
            The cost function to optimize
        threshold: float
            The threshold to consider that the value is optimal

        Returns
        -------
        The optimized value
        """

        min_so_far = lbx
        max_so_far = ubx

        while True:
            # Get the cost function
            x = (min_so_far + max_so_far) / 2
            value = cost_function(x)

            # If the value is within the threshold, we are done
            if -threshold / 2 < value < threshold / 2 or np.abs(min_so_far - max_so_far) < threshold:
                break

            if value > 0:
                # If the value is positive, we need to increase the minimum value so far
                min_so_far = x
            else:
                # If the value is negative, we need to decrease the maximum value so far
                max_so_far = x

        return x
