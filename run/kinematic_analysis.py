import numpy as np

from shoulder import ModelBiorbd, ModelMujoco, Plotter, Animater, ControlsTypes, IntegrationMethods


def main():
    # Setup
    tf = 50  # seconds
    control_type = ControlsTypes.EMG
    show_animate = False
    show_graphs = True

    # Aliases
    models = (
        (ModelMujoco("models/arm26.xml"), IntegrationMethods.RK4),
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
        q_integrated, qdot_integrated = model.integrate(
            t=t,
            states=np.concatenate((q, qdot)),
            controls=controls,
            controls_type=control_type,
            integration_method=integration_method,
        )

        # Visualize
        if show_animate:
            Animater(model, q_integrated).show()

        if show_graphs:
            plotter = Plotter(t=t, model=model, q=q_integrated, qdot=qdot_integrated, tau=tau, emg=emg)
            plotter.plot_movement()

    plotter.show()


if __name__ == "__main__":
    main()
