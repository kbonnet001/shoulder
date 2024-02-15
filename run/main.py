from enum import Enum
from functools import partial

import numpy as np
from scipy import integrate

from shoulder import Model, animate, plot_movement, plot_com, show


class Controls(Enum):
    EMG = 0
    TORQUE = 1


def main():
    # Setup
    tf = 50  # seconds
    control = Controls.TORQUE
    show_animate = True
    graphs = (plot_movement,)  # plot_com

    # Aliases
    model = Model("models/Wu.bioMod")
    n_q = model.n_q
    n_muscles = model.n_muscles

    # Prepare the states
    q = np.zeros((n_q,))
    qdot = np.zeros((n_q,))

    # Prepare controls
    tau = None
    emg = None
    if control == Controls.EMG:
        emg = np.zeros((n_muscles,))
        func = partial(model.forward_dynamics_muscles, emg=emg)
    elif control == Controls.TORQUE:
        tau = np.zeros((n_q,))
        func = partial(model.forward_dynamics, tau=tau)
    else:
        raise NotImplementedError(f"Control {control} not implemented")

    # Integrate
    t = np.linspace(0, tf, tf * 100)  # 100 Hz
    integrated = integrate.solve_ivp(
        fun=func,
        t_span=(0, tf),
        y0=np.concatenate((q, qdot)),
        method="RK45",
        t_eval=t,
    )
    q_integrated = integrated.y[: model.n_q, :]
    qdot_integrated = integrated.y[model.n_q :, :]

    # Visualize
    if show_animate:
        animate(q_integrated, model)

    for graph in graphs:
        graph(t=t, model=model, q=q_integrated, qdot=qdot_integrated, tau=tau, emg=emg)
    show()


if __name__ == "__main__":
    main()
