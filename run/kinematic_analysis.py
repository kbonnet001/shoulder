import numpy as np

from shoulder import ModelBiorbd, ModelMujoco, Plotter, ControlsTypes, IntegrationMethods


def main():
    # Setup
    tf = 50  # seconds
    control_type = ControlsTypes.EMG
    show_animate = False
    show_graphs = True

    # Aliases
    models = (
        (ModelMujoco("../external/myo_sim/arm/myoarm.xml"), IntegrationMethods.RK4),
        (ModelBiorbd("models/Wu_Thelen.bioMod"), IntegrationMethods.RK45),
    )

    for model, integration_method in models:
        n_q = model.n_q
        n_muscles = model.n_muscles

        # Prepare the states
        q = np.zeros((n_q,))
        qdot = np.zeros((n_q,))

        # Prepare controls
        tau = None
        emg = None
        if control_type == ControlsTypes.EMG:
            controls = np.ones((n_muscles,))
            emg = controls
        elif control_type == ControlsTypes.TORQUE:
            controls = np.zeros((n_q,))
            tau = controls
        else:
            raise NotImplementedError(f"Control {control_type} not implemented")

        # Integrate
        t = np.linspace(0, tf, tf * 100)  # 100 Hz
        states = model.integrate(
            t=t,
            states=np.concatenate((q, qdot)),
            controls=controls,
            controls_type=control_type,
            integration_method=integration_method,
        )

        # Visualize
        if show_animate:
            model.animate(states)

        if show_graphs:
            q_integrated = states[0]
            qdot_integrated = states[1]
            plotter = Plotter(t=t, model=model, q=q_integrated, qdot=qdot_integrated, tau=tau, emg=emg)
            plotter.plot_movement()

    plotter.show()


if __name__ == "__main__":
    main()
