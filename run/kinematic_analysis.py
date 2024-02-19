from enum import Enum
from functools import partial

import numpy as np
from scipy import integrate

from shoulder import Model, Plotter, Animater


class Controls(Enum):
    EMG = 0
    TORQUE = 1


def main():
    # Setup
    tf = 50  # seconds
    control = Controls.EMG
    show_animate = False
    show_graphs = True

    # Aliases
    model = Model("models/Wu_Thelen.bioMod")
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
        Animater(model, q_integrated).show()

    if show_graphs:
        Plotter(t=t, model=model, q=q_integrated, qdot=qdot_integrated, tau=tau, emg=emg).show()


if __name__ == "__main__":
    main()
