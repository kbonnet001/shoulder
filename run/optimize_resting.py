import time

import casadi
import numpy as np
from shoulder import ModelBiorbd, ControlsTypes, MuscleHelpers, MuscleParameter


# Add muscletendon equilibrium constraint
# Multivariate normal => center + noise , returns covariance matrix (Robust optimization)
# TODO The tendon slack length should be opimized at optimal length instead of shortest length?
# OR
# TODO use tendon force if flpe == 0?
# TODO Does the model target the same pose as the astronaut?


class Results:
    class Values:
        def __init__(self, values: np.ndarray, reference: np.ndarray, x0: np.ndarray, lb: np.ndarray, ub: np.ndarray):
            if len(values.shape) > 1:
                if values.shape[1] != 1:
                    raise ValueError("Values must be a column vector")
                values = values[:, 0]
            if len(reference.shape) > 1:
                if reference.shape[1] != 1:
                    raise ValueError("Reference must be a column vector")
                reference = reference[:, 0]
            if len(x0.shape) > 1:
                if x0.shape[1] != 1:
                    raise ValueError("Initial guess must be a column vector")
                x0 = x0[:, 0]
            if len(lb.shape) > 1:
                if lb.shape[1] != 1:
                    raise ValueError("Lower bounds must be a column vector")
                lb = lb[:, 0]
            if len(ub.shape) > 1:
                if ub.shape[1] != 1:
                    raise ValueError("Upper bounds must be a column vector")
                ub = ub[:, 0]

            self.values = values
            self.reference = reference
            self.x0 = x0
            self.lb = lb
            self.ub = ub

        def __getitem__(self, item: int) -> float:
            return Results.Values(
                np.array([self.values[item]]),
                np.array([self.reference[item]]),
                np.array([self.x0[item]]),
                np.array([self.lb[item]]),
                np.array([self.ub[item]]),
            )

        def __str__(self) -> str:
            s = ""
            for i in range(len(self.values)):
                diff = self.values[i] - self.reference[i]
                s += (
                    f"{self.lb[i]:>4.1f} <= {self.values[i]:>4.1f} ({self.x0[i]:>4.1f}) <= {self.ub[i]:>4.1f} "
                    f"/ {self.reference[i]:>4.1f} ({'+' if diff > 0 else '-'} {np.abs(diff):>4.1f})"
                )
            return s

    def __init__(
        self,
        model: ModelBiorbd,
        optimal_lengths: Values,
        tendon_slack_lengths: Values,
    ):
        self.model = model

        self.optimal_lengths = optimal_lengths
        self.tendon_slack_lengths = tendon_slack_lengths

    def __str__(self) -> str:
        s = f"Model: {self.model.name}\n"
        for i in range(self.model.n_muscles):
            s += f"  {self.model.muscle_names[i]}:\n"
            s += f"    {'Optimal length:':<30}{self.optimal_lengths[i]}\n"
            s += f"    {'Tendon slack length:':<30}{self.tendon_slack_lengths[i]}\n"
        return s


def optimize_muscle_parameters(
    cx, model: ModelBiorbd, use_predefined_muscle_ratio_values, expand: bool = True
) -> Results:
    """
    Find values for the tendon slack lengths that do not produce any muscle force

    Parameters
    ----------
    cx: casadi.SX
        The casadi symbol
    model: ModelBiorbd
        The model to optimize
    use_predefined_muscle_ratio_values: bool
        If True, the muscle parameters use predefined values to compute their initial guess
    expand: bool
        If True, the casadi functions are expanded

    Returns
    -------
    Results
        The results of the optimization
    """
    start = time.time()
    print(f"Optimizing muscle parameters for model {model.name}...")

    # Declare some aliases
    n_muscles = model.n_muscles

    # Backup the predefined muscle parameters
    reference_optimal_lengths = [
        float(model.get_muscle_parameter(i, MuscleParameter.OPTIMAL_LENGTH)) for i in range(n_muscles)
    ]
    reference_tendon_slack_lengths = [
        float(model.get_muscle_parameter(i, MuscleParameter.TENDON_SLACK_LENGTH)) for i in range(n_muscles)
    ]

    # Set the model to an initial guess based on the geometric properties of the muscles
    for i in range(n_muscles):
        q = model.strongest_poses[model.muscle_names[i]]
        pennation_angle = model.get_muscle_parameter(i, MuscleParameter.PENNATION_ANGLE)
        muscle_tendon_length = model.muscles_kinematics(q, muscle_index=i) * np.cos(pennation_angle)

        if use_predefined_muscle_ratio_values:
            muscle_to_tendon_length_ratio = reference_tendon_slack_lengths[i] / reference_optimal_lengths[i]
        else:
            muscle_to_tendon_length_ratio = 0.4

        tendon_length = muscle_tendon_length * muscle_to_tendon_length_ratio
        fiber_length = (muscle_tendon_length - tendon_length) / np.cos(pennation_angle)
        model.set_muscle_parameters(index=i, tendon_slack_length=tendon_length, optimal_length=fiber_length)

    # Prepare the decision variables
    optimal_lengths_cx = cx.sym("optimal_lengths", n_muscles, 1)
    tendon_slack_lengths_cx = cx.sym("tendon_slack_lengths", n_muscles, 1)
    optimal_lengths_mx = casadi.MX.sym("optimal_lengths_mx", n_muscles, 1)
    tendon_slack_lengths_mx = casadi.MX.sym("tendon_slack_lengths_mx", n_muscles, 1)

    # Find the initial guess for the tendon slack length by first finding a somewhat okay optimal length found by
    # maximizing force at the strongest pose, using the predefined tendon slack length
    print("Pre-optimizing the tendon slack lengths...")
    strongest_poses = model.strongest_poses
    optimal_lengths_at_strongest = MuscleHelpers.find_optimal_length(model, all_poses=strongest_poses, expand=expand)
    for i in range(n_muscles):
        model.set_muscle_parameters(index=i, optimal_length=optimal_lengths_at_strongest[i])
    tendon_slack_lengths_x0 = MuscleHelpers.find_minimal_tendon_slack_lengths(
        model, all_poses=strongest_poses, expand=expand
    )
    # Ensure the tendon slack lengths are not too small (minimum initial guess 5mm)
    tendon_slack_lengths_x0[tendon_slack_lengths_x0 < 0.005] = 0.005

    # Now reoptimize the optimal lengths assuming the tendon slack lengths are the ones found")
    print("Pre-optimizing the optimal lengths...")
    for i in range(n_muscles):
        model.set_muscle_parameters(index=i, tendon_slack_length=tendon_slack_lengths_x0[i])
    optimal_lengths_x0 = MuscleHelpers.find_optimal_length(model, all_poses=strongest_poses, expand=expand)

    # Declare the bounds of the optimization problem
    optimal_lengths_lb = optimal_lengths_x0 * 0.5
    optimal_lengths_ub = optimal_lengths_x0 * 2.0
    tendon_slack_lengths_lb = tendon_slack_lengths_x0 * 0.5
    tendon_slack_lengths_ub = tendon_slack_lengths_x0 * 2.0

    # Merge the decision variables to a single vector
    x = casadi.vertcat(optimal_lengths_cx, tendon_slack_lengths_cx)
    x_mx = casadi.vertcat(optimal_lengths_mx, tendon_slack_lengths_mx)
    x0 = casadi.vertcat(optimal_lengths_x0, tendon_slack_lengths_x0)
    lbx = casadi.vertcat(optimal_lengths_lb, tendon_slack_lengths_lb)
    ubx = casadi.vertcat(optimal_lengths_ub, tendon_slack_lengths_ub)

    # Set the muscle parameters to the decision variables
    for i in range(n_muscles):
        model.set_muscle_parameters(
            index=i, optimal_length=optimal_lengths_mx[i], tendon_slack_length=tendon_slack_lengths_mx[i]
        )

    # Declare the cost functions
    print("Declaring the cost functions and constraints...")
    f_mx = []
    g_mx = []
    lbg = []
    ubg = []

    # Acceleration should be zero at the relaxed pose
    q = casadi.MX(model.relaxed_pose)
    qdot = casadi.MX.zeros(model.n_q, 1)
    emg = casadi.MX.ones(n_muscles, 1) * 0.01
    qddot = model.forward_dynamics(q=q, qdot=qdot, controls=emg, controls_type=ControlsTypes.EMG)[1]
    f_mx.append(qddot**2)

    emg = casadi.MX.ones(n_muscles, 1)
    qdot = casadi.MX.zeros(model.n_q, 1)
    for i in range(n_muscles):
        q_optimal = casadi.MX(strongest_poses[model.muscle_names[i]])
        flpe, flce = model.muscle_force_coefficients(emg, q_optimal, muscle_index=i)
        force = model.muscle_force(emg, q_optimal, qdot, muscle_index=i)

        # Forces should be maximized at the optimal pose
        f_mx.append(1000 * -(force**2))

        # Forces should be maximized at the optimal pose
        g_mx.append(flce)
        lbg.append(0.95)
        ubg.append(1.00)

        # Passive force should be almost zero at the optimal pose
        g_mx.append(flpe)
        lbg.append(0.000)
        ubg.append(0.001)

    # Converting to casadi functions
    if len(f_mx) == 0:
        f = []
    else:
        f = casadi.Function("f", [x_mx], [casadi.sum1(casadi.vertcat(*f_mx))])
        if expand:
            f = f.expand()
        f = f(x)

    if len(g_mx) == 0:
        g = []
    else:
        g = casadi.Function("g", [x_mx], [casadi.vertcat(*g_mx)])
        if expand:
            g = g.expand()
        g = g(x)
        lbg = np.array(lbg)
        ubg = np.array(ubg)

    # Solve the optimization problem
    print("Solving the optimization problem...")
    solver = casadi.nlpsol(
        "solver",
        "ipopt",
        {"x": x, "f": f, "g": g},
        {"ipopt.max_iter": 10000, "ipopt.hessian_approximation": "limited-memory"},
    )
    sol = solver(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
    print(f"Optimization done in {time.time() - start:.2f} s")

    # Set the muscle parameters to the decision variables
    for i in range(n_muscles):
        model.set_muscle_parameters(
            index=i, optimal_length=float(sol["x"][:n_muscles][i]), tendon_slack_length=float(sol["x"][n_muscles:][i])
        )

    for i in range(n_muscles):
        q_optimal = strongest_poses[model.muscle_names[i]]
        flpe, flce = model.muscle_force_coefficients(np.ones((n_muscles, 1)), np.array(q_optimal), muscle_index=i)
        print(f"{model.muscle_names[i]}: flpe = {float(flpe)}, flce = {float(flce)}")
    print(f"qddot = {np.sum(casadi.Function('qddot', [x_mx], [qddot])(sol['x']))}")

    # Parse the results
    return Results(
        model=model,
        optimal_lengths=Results.Values(
            values=np.array(sol["x"][:n_muscles]) * 100,
            reference=np.array(reference_optimal_lengths) * 100,
            x0=np.array(optimal_lengths_x0) * 100,
            lb=np.array(optimal_lengths_lb) * 100,
            ub=np.array(optimal_lengths_ub) * 100,
        ),
        tendon_slack_lengths=Results.Values(
            values=np.array(sol["x"][n_muscles:]) * 100,
            reference=np.array(reference_tendon_slack_lengths) * 100,
            x0=np.array(tendon_slack_lengths_x0) * 100,
            lb=np.array(tendon_slack_lengths_lb) * 100,
            ub=np.array(tendon_slack_lengths_ub) * 100,
        ),
    )


def main():
    # Aliases
    cx = casadi.SX
    use_predefined_muscle_ratio_values = True
    models = (
        ModelBiorbd("models/Wu_DeGroote.bioMod", use_casadi=True),
        # ModelBiorbd("models/Wu_Thelen.bioMod", use_casadi=True),
    )

    results = []
    for model in models:
        # Optimize
        results.append(optimize_muscle_parameters(cx, model, use_predefined_muscle_ratio_values, expand=True))

    for result in results:
        print(result)


if __name__ == "__main__":
    main()
