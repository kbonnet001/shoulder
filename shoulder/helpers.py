from typing import Callable

import numpy as np


class OptimizationHelpers:
    @staticmethod
    def simple_gradient_descent(x0: float, cost_function_jacobian: Callable, threshold: float = 1e-8):
        """
        Simple gradient descent algorithm, ifthe algorithm is stuck bouncing between two values, we increase the bouncing
        modifier which reduces the step size. This works makes the assumption that the function only has one local
        minimum which is the global minimum

        Parameters
        ----------
        x0: float
            The initial guess
        cost_function_jacobian: Callable
            The jacobian function to optimize
        threshold: float
            The threshold to consider that the muscle is producing maximal force

        Returns
        -------
        The optimized value
        """

        if x0 == 0:
            raise ValueError("The initial guess cannot be 0")

        previous_sign = 1
        nb_sign_bouncing = 0
        x = x0
        while True:
            # Get the muscle force
            jaco_value = cost_function_jacobian(x)

            # If the change is smaller than the threshold, we are at an optimal value
            if np.sum(np.abs(jaco_value)) < threshold:
                return x

            # This bouncing mechanism is used to avoid the optimizer to get stuck bouncing between two values around
            # the optimal value
            if previous_sign * jaco_value < 0:
                nb_sign_bouncing += 1
            previous_sign = np.sign(jaco_value)
            bouncing_modifier = 10 ** (-nb_sign_bouncing)

            x += min(x * np.clip(jaco_value, -0.1 * bouncing_modifier, 0.1 * bouncing_modifier))

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
            if -threshold / 2 < value < threshold / 2:
                return x

            if value > 0:
                # If the value is positive, we need to increase the minimum value so far
                min_so_far = x
            else:
                # If the value is negative, we need to decrease the maximum value so far
                max_so_far = x
