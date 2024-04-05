import casadi
import numpy as np
from shoulder import ModelBiorbd, ControlsTypes, MuscleParameter, OptimizationHelpers


# Add muscletendon equilibrium constraint
# Multivariate normal => center + noise , returns covariance matrix


def find_optimal_length_assuming_no_tendon(model: ModelBiorbd) -> np.array:
    """
    Find values for the optimal muscle lengths where each muscle produces maximal force at their respective strongest
    pose. We could use IPOPT for that, but since the initial guess is so poor, sometimes we get to 0N force, which
    confuses the optimizer. This function uses a custom gradient descent algorithm that is more robust to this
    kind of problem.

    The method resets the model to its original state after the optimization

    Parameters
    ----------
    model: ModelBiorbd
        The model to use

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
    ).expand()

    # Optimize for each muscle
    x = np.ndarray(n_muscles) * np.nan
    for i in range(n_muscles):
        # Evaluate the jacobian at q
        q = model.strongest_poses[model.muscle_names[i]]
        jaco = casadi.Function(
            "jacobian_at_q",
            [optimal_lengths_mx],
            [muscle_forces_coefficients_jacobian(optimal_lengths=optimal_lengths_mx, q=q)["jacobian"][i]],
            ["optimal_lengths"],
            ["jacobian_at_q"],
        ).expand()

        # Optimize for the current muscle
        flce_initial_guess = model.muscles_kinematics(q)[i, 0] / 2
        x[i] = OptimizationHelpers.simple_gradient_descent(x0=flce_initial_guess, cost_function_jacobian=jaco)

    # Set back the original optimal lengths
    for i in range(n_muscles):
        model.set_muscle_parameters(index=i, optimal_length=optimal_lengths_bak[i])

    return x


def find_minimal_tendon_slack_lengths(model: ModelBiorbd, emg: np.ndarray, q: np.array, qdot: np.array) -> np.array:
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
    ).expand()

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
        ).expand()
        x[i] = OptimizationHelpers.squeezing_optimization(lbx=0, ubx=1, cost_function=force)

    # Set back the original tendon slack lengths
    for i in range(n_muscles):
        model.set_muscle_parameters(index=i, tendon_slack_length=tendon_slack_lengths_bak[i])

    return x


def optimize_tendon_slack_lengths(
    cx, model: ModelBiorbd, emg: np.ndarray, q: np.ndarray, qdot: np.ndarray
) -> np.ndarray:
    """
    Find values for the tendon slack lengths that do not produce any muscle force
    """

    # Declare some aliases
    n_muscles = model.n_muscles

    # Prepare the decision variables

    # Find and set  initial guess for the optimal muscle lengths that is highest at the strongest pose
    optimal_lengths_cx = cx.sym("optimal_lengths", n_muscles, 1)
    optimal_lengths_mx = casadi.MX.sym("optimal_lengths_mx", n_muscles, 1)
    optimal_lengths_x0 = find_optimal_length_assuming_no_tendon(model)
    optimal_lengths_lb = optimal_lengths_x0 * 0.5
    optimal_lengths_ub = optimal_lengths_x0 * 1.5

    # Set the model to the optimal muscle lengths initial guess
    for i in range(n_muscles):
        model.set_muscle_parameters(index=i, optimal_length=optimal_lengths_x0[i])

    # Get initial guesses and bounds based on the model
    tendon_slack_lengths_cx = cx.sym("tendon_slack_lengths", n_muscles, 1)
    tendon_slack_lengths_mx = casadi.MX.sym("tendon_slack_lengths_mx", n_muscles, 1)
    tendon_slack_lengths_x0 = find_minimal_tendon_slack_lengths(model, emg, q, qdot)
    tendon_slack_lengths_lb = tendon_slack_lengths_x0 * 0.9
    tendon_slack_lengths_ub = tendon_slack_lengths_lb * 1.1

    # Set arbitrary initial guess
    x = casadi.vertcat(optimal_lengths_cx, tendon_slack_lengths_cx)
    x_mx = casadi.vertcat(optimal_lengths_mx, tendon_slack_lengths_mx)
    lbx = casadi.vertcat(optimal_lengths_lb, tendon_slack_lengths_lb)
    ubx = casadi.vertcat(optimal_lengths_ub, tendon_slack_lengths_ub)
    x0 = casadi.vertcat((optimal_lengths_x0, tendon_slack_lengths_x0))

    # Compute the cost functions
    f_mx = []
    for i in range(n_muscles):
        model.set_muscle_parameters(
            index=i, optimal_length=optimal_lengths_mx[i], tendon_slack_length=tendon_slack_lengths_mx[i]
        )
    f_mx.append(
        model.forward_dynamics(
            q=casadi.MX(q), qdot=casadi.MX(qdot), controls=casadi.MX(emg), controls_type=ControlsTypes.EMG
        )[1]
        ** 2
    )

    # Compute some non-linear constraints
    g_mx = []
    lbg = np.array([])
    ubg = np.array([])

    # Solve the program
    f = casadi.Function("f", [x_mx], [casadi.sum1(casadi.vertcat(*f_mx))]).expand()(x)
    g = casadi.Function("g", [x_mx], [casadi.vertcat(*g_mx)]).expand()(x)
    solver = casadi.nlpsol("solver", "ipopt", {"x": x, "f": f, "g": g})
    sol = solver(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)

    return sol["x"]


def main():
    # Aliases
    cx = casadi.SX
    models = (ModelBiorbd("models/Wu_DeGroote.bioMod", use_casadi=True),)

    for model in models:
        n_q = model.n_q
        n_muscles = model.n_muscles
        q = np.ones(n_q) * 0.001
        qdot = np.zeros(n_q)
        emg = np.ones(n_muscles) * 0.01

        opimized_tendon_slack_lengths = optimize_tendon_slack_lengths(cx, model, emg, q, qdot)

        # Print the results
        print(f"The optimal tendon slack lengths are: {opimized_tendon_slack_lengths}")


if __name__ == "__main__":
    main()
