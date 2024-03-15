import casadi
import numpy as np
from shoulder import ModelBiorbd, ControlsTypes


def compute_qdot(x: casadi.MX | casadi.SX, model: ModelBiorbd, q: np.ndarray, qdot: np.ndarray, emg: np.ndarray):
    n_q = model.n_q
    n_muscles = model.n_muscles

    # Compute the dynamics using the MX as biorbd is MX based
    optimal_lengths_mx = casadi.MX.sym("optimalLength", n_muscles, 1)
    q_mx = casadi.MX.sym("q", n_q, 1)
    qdot_mx = casadi.MX.sym("qdot", n_q, 1)
    emg_mx = casadi.MX.sym("emg", n_muscles, 1)
    for i in range(n_muscles):
        model._model.muscle(i).characteristics().setOptimalLength(optimal_lengths_mx[i])  # TODO: Add interface for this
    xdot = model.forward_dynamics(q=q_mx, qdot=qdot_mx, controls=emg_mx, controls_type=ControlsTypes.EMG)

    # Convert the outputs to the type corresponding to the x vector and collapsing the graph at q, qdot and emg
    xdot = casadi.Function(
        "xdot",
        [q_mx, qdot_mx, emg_mx, optimal_lengths_mx],
        xdot,
        ["q_in", "qdot_in", "emg_in", "optimal_length_in"],
        ["q", "qdot"],
    )
    qdot = xdot(q_in=q, qdot_in=qdot, emg_in=emg, optimal_length_in=x)["qdot"]

    # Return qdot
    return qdot


def main():
    # Aliases
    cx = casadi.MX
    models = (ModelBiorbd("models/Wu_Thelen.bioMod", use_casadi=True),)

    for model in models:
        # Initialize the model
        n_q = model.n_q
        n_muscles = model.n_muscles
        q = np.zeros((n_q,))
        qdot = np.zeros((n_q,))
        emg = np.ones((n_muscles,)) * 0.0

        # Prepare the decision variables
        x = cx.sym("optimalLengths", n_muscles, 1)

        # Get initial guesses and bounds based on the model
        x0 = np.array(
            [float(model._model.muscle(i).characteristics().optimalLength().to_mx()) for i in range(n_muscles)]
        )
        lbx = x0 * 0.8
        ubx = x0 * 1.2

        # Compute the cost functions
        obj = casadi.sum1(compute_qdot(x, model, q, qdot, emg) ** 2)

        # Compute some non-linear constraints
        g = cx()
        lbg = np.array([])
        ubg = np.array([])

        # Solve the program
        solver = casadi.nlpsol("solver", "ipopt", {"x": x, "f": obj, "g": g})
        sol = solver(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)

        # Print the results
        print(f"The optimal lengths are: {sol['x']}")
        print(f"The optimal cost is: {sol['f']}")


if __name__ == "__main__":
    main()
