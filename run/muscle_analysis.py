from enum import Enum, auto

import numpy as np
from matplotlib import pyplot as plt
import shoulder

"""
Potential bug found in the code
    - The force-velocity relationship is opposite (concenctric vs eccentric) in the Thelen and DeGroote models
        - EDIT: This is actually fine as the model is based on + being eccentric instead of concentric

    - IN BIOVIZ : THE MUSCLE ARE BADLY CAST
"""


def ease_in_out(start: float, end: float, n_points: int):
    t = np.linspace(0, 1, n_points)
    return start + (end - start) * ((t < 0.5) * (4 * t * t * t) + (t >= 0.5) * (1 - np.power(-2 * t + 2, 3) / 2))


class XAxis(Enum):
    TIME = auto()
    MUSCLE_PARAMETERS = auto()
    KINEMATICS = auto()


def main():
    # Setup
    tf = 4  # second
    frequency = 50  # Hz
    dof_index = 0
    muscle_index = 0
    show_animate = False
    show_graphs = True
    x_axes = [XAxis.TIME, XAxis.MUSCLE_PARAMETERS, XAxis.KINEMATICS]

    # Create a time vector
    n_points = tf * frequency
    t = np.linspace(0, tf, n_points)

    # Evaluate from the model
    models = [shoulder.Model("models/Wu_Thelen.bioMod"), shoulder.Model("models/Wu_DeGroote.bioMod")]
    model_names = ["Thelen", "DeGroote"]
    model_colors = ["b", "r"]
    model = models[0]  # Alias for the computation which are not muscle models specific
    q = np.zeros((model.n_q, n_points))

    # Elevate the arm for 1 second, then lower it for 1 second
    q[dof_index, :] = np.concatenate(
        (ease_in_out(0, -np.pi / 2, int(n_points / 2)), ease_in_out(-np.pi / 2, 0, int(n_points / 2)))
    )

    # compute qdot as the derivative of q in a non-phase shifted way
    qdot = np.ndarray((model.n_q, n_points))
    qdot[:, 1:-1] = (q[:, 2:] - q[:, :-2]) / (t[2:] - t[:-2])
    qdot[:, [0, -1]] = qdot[:, [1, -2]]

    length, velocity = model.muscles_kinematics(q, qdot)
    length = length[muscle_index, :]
    velocity = velocity[muscle_index, :]
    emg = np.ones((model.n_muscles, n_points))

    # Compute muscle force coefficients
    flce = []
    fvce = []
    for model in models:
        flce_model, fvce_model = model.muscle_force_coefficients(emg, q, qdot, muscle_index=muscle_index)
        flce.append(flce_model)
        fvce.append(fvce_model)

    # Animate
    if show_animate:
        shoulder.show()

    # Plot them
    if show_graphs:
        for x_axis in x_axes:
            if x_axis == XAxis.TIME:
                x1 = x2 = t
                x1_label = x2_label = "Time (s)"
            elif x_axis == XAxis.MUSCLE_PARAMETERS:
                x1, x1_label = length, "Muscle length"
                x2, x2_label = velocity, "Muscle velocity"
            elif x_axis == XAxis.KINEMATICS:
                x1, x1_label = q[dof_index, :], "q"
                x2, x2_label = qdot[dof_index, :], "qdot"
            else:
                raise NotImplementedError(f"X axis {x_axis} not implemented")

            shoulder.plot_muscle_force_coefficients(
                title=f"Force-Length relationship ({x_axis.name})",
                x=x1,
                x_label=x1_label,
                y_left=flce,
                y_left_label=model_names,
                y_left_colors=model_colors,
                y_right=length,
                y_right_label="Muscle length",
                y_right_color="g",
            )

            shoulder.plot_muscle_force_coefficients(
                title=f"Force-Velocity relationship ({x_axis.name})",
                x=x2,
                x_label=x2_label,
                y_left=fvce,
                y_left_label=model_names,
                y_left_colors=model_colors,
                y_right=velocity,
                y_right_label="Muscle velocity",
                y_right_color="g",
            )

        shoulder.plot_muscle_force_coefficients_surface(
            models=models, q=q, qdot=qdot, emg=emg, muscle_index=muscle_index, dof_index=dof_index
        )

        plt.show()


if __name__ == "__main__":
    main()
