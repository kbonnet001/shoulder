import time

import casadi
import numpy as np
from shoulder import ModelBiorbd, ControlsTypes, MuscleHelpers


# Add muscletendon equilibrium constraint
# Multivariate normal => center + noise , returns covariance matrix

class Results:
    def __init__(self, model: ModelBiorbd, optimal_lengths: np.ndarray, tendon_slack_lengths: np.ndarray):
        self.model = model
        self.optimal_lengths = optimal_lengths
        self.tendon_slack_lengths = tendon_slack_lengths

    def __str__(self) -> str:
        s = f"Model: {self.model.name}\n"
        for i in range(self.model.n_muscles):
            s += f"  {self.model.muscle_names[i]}:\n"
            s += f"    {"Optimal length:":<20}{float(self.optimal_lengths[i]):>6,.3f}\n"
            s += f"    {"Tendon slack length:":<20}{float(self.tendon_slack_lengths[i]):>6,.3f}\n"
        return s


def optimize_muscle_parameters(cx, model: ModelBiorbd, emg: np.ndarray, q: np.ndarray, qdot: np.ndarray, expand: bool = True) -> Results:
    """
    Find values for the tendon slack lengths that do not produce any muscle force
    """
    start = time.time()
    print(f"Optimizing muscle parameters for model {model.name}...")

    # Declare some aliases
    n_muscles = model.n_muscles

    # Prepare the decision variables
    optimal_lengths_cx = cx.sym("optimal_lengths", n_muscles, 1)
    tendon_slack_lengths_cx = cx.sym("tendon_slack_lengths", n_muscles, 1)
    optimal_lengths_mx = casadi.MX.sym("optimal_lengths_mx", n_muscles, 1)
    tendon_slack_lengths_mx = casadi.MX.sym("tendon_slack_lengths_mx", n_muscles, 1)

    # Find the initial guess for the tendon slack length by first finding a somewhat okay optimal length found by
    # maximizing force at the strongest pose, using the predefined tendon slack length
    print("Pre-optimizing the tendon slack lengths...")
    optimal_lengths_at_strongest = MuscleHelpers.find_optimal_length_assuming_strongest_pose(model, expand=expand)
    for i in range(n_muscles):
        model.set_muscle_parameters(index=i, optimal_length=optimal_lengths_at_strongest[i])
    tendon_slack_lengths_x0 = MuscleHelpers.find_minimal_tendon_slack_lengths(model, emg, q, qdot, expand=expand)

    # Now reoptimize the optimal lengths assuming the tendon slack lengths are the ones found")
    print("Pre-optimizing the optimal lengths...")
    for i in range(n_muscles):
        model.set_muscle_parameters(index=i, tendon_slack_length=tendon_slack_lengths_x0[i])
    optimal_lengths_x0 = MuscleHelpers.find_optimal_length_assuming_strongest_pose(model, expand=expand)

    # Declare the bounds of the optimization problem
    optimal_lengths_lb = optimal_lengths_x0 * 0.5
    optimal_lengths_ub = optimal_lengths_x0 * 1.5
    tendon_slack_lengths_lb = tendon_slack_lengths_x0 * 0.9
    tendon_slack_lengths_ub = tendon_slack_lengths_lb * 1.1

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
    print("Declaring the cost functions...")
    f_mx = []
    f_mx.append(
        model.forward_dynamics(
            q=casadi.MX(q), qdot=casadi.MX(qdot), controls=casadi.MX(emg), controls_type=ControlsTypes.EMG
        )[1]
        ** 2
    )

    f = casadi.Function("f", [x_mx], [casadi.sum1(casadi.vertcat(*f_mx))])
    if expand:
        f = f.expand()
    f = f(x)

    # Declare the constraints
    print("Declaring the constraints...")
    g_mx = []
    lbg = np.array([])
    ubg = np.array([])

    g = casadi.Function("g", [x_mx], [casadi.vertcat(*g_mx)])
    if expand:
        g = g.expand()
    g = g(x)

    # Solve the optimization problem
    print("Solving the optimization problem...")
    solver = casadi.nlpsol("solver", "ipopt", {"x": x, "f": f, "g": g})
    sol = solver(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
    print(f"Optimization done in {time.time() - start:.2f} s")

    # Parse the results
    return Results(model=model, optimal_lengths=sol["x"][:n_muscles], tendon_slack_lengths=sol["x"][n_muscles:])


def main():
    # Aliases
    cx = casadi.SX
    models = (
        ModelBiorbd("models/Wu_DeGroote.bioMod", use_casadi=True),
        ModelBiorbd("models/Wu_Thelen.bioMod", use_casadi=True),
    )

    results = []
    for model in models:
        n_q = model.n_q
        n_muscles = model.n_muscles
        q = np.ones(n_q) * 0.001
        qdot = np.zeros(n_q)
        emg = np.ones(n_muscles) * 0.01

        # Optimize
        results.append(optimize_muscle_parameters(cx, model, emg, q, qdot, expand=True))

    for result in results:
        print(result)


if __name__ == "__main__":
    main()
