import numpy as np
import shoulder

"""
Notes
    - Bug: in the force-velocity relationship is opposite (concenctric vs eccentric) in the Thelen and DeGroote models
        - EDIT: This is actually fine as the model is based on + being eccentric instead of concentric
    - Bug: Thelen does not produce any force in concentric
    

    - bug IN BIOVIZ : THE MUSCLE ARE BADLY CAST
"""


def ease_in_out(start: float, end: float, n_points: int):
    t = np.linspace(0, 1, n_points)
    return start + (end - start) * ((t < 0.5) * (4 * t * t * t) + (t >= 0.5) * (1 - np.power(-2 * t + 2, 3) / 2))


def main():
    # Setup
    tf = 0.4  # second
    frequency = 50  # Hz
    show_animate = False
    show_graphs = True
    coefficients = shoulder.MuscleCoefficientPlots.NONE
    surface = shoulder.MuscleSurfacePlots.FORCE

    # Create a time vector
    n_points = int(tf * frequency)
    if n_points != tf * frequency:
        raise ValueError("tf must be a multiple of 1/frequency")
    t = np.linspace(0, tf, n_points)

    # Evaluate from the model
    models = [
        (shoulder.ModelBiorbd("models/Wu_Thelen.bioMod"), 0, 0),
        (shoulder.ModelBiorbd("models/Wu_DeGroote.bioMod"), 0, 0),
        (shoulder.ModelMujoco("../external/myo_sim/arm/myoarm.xml"), 4, 0),
    ]
    model_colors = ["b", "r", "m"]

    fig = None
    for i_model, (model, dof_index, muscle_index) in enumerate(models):
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

        # Animate
        if show_animate:
            model.animate(states=[q, qdot])

        # Plot them
        if show_graphs:
            plotter = shoulder.Plotter(
                model=model, t=t, q=q, qdot=qdot, emg=emg, muscle_index=muscle_index, dof_index=dof_index
            )
            fig = plotter.plot_muscle_force_coefficients(plots=coefficients, color=model_colors[i_model], fig=fig)
            plotter.plot_muscle_surface(plots=surface, axis_id=100 + len(models) * 10 + i_model + 1)

    if show_graphs:
        plotter.show()


if __name__ == "__main__":
    main()
