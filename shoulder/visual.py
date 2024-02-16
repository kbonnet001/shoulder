from enum import Enum, auto

import bioviz
import numpy as np
from matplotlib import pyplot as plt

from .model import Model


def animate(model: Model, q: np.ndarray):
    viz = bioviz.Viz(
        loaded_model=model.model,
        show_local_ref_frame=False,
        show_segments_center_of_mass=False,
        show_global_center_of_mass=False,
        show_gravity_vector=False,
    )
    viz.load_movement(q)
    viz.set_camera_roll(np.pi / 2)
    viz.exec()


def plot_muscle_force_coefficients(
    title: str,
    x: np.ndarray,
    x_label: str,
    y_left: list[np.ndarray],
    y_left_label: list[str],
    y_left_colors: list[str],
    y_right: np.ndarray = None,
    y_right_label: str = None,
    y_right_color: str = None,
):
    plt.figure(title)
    n_points = x.shape[0]

    for i in range(n_points):
        for j in range(len(y_left)):
            label = None
            if i == n_points - 1:
                label = y_left_label[j]
            plt.plot(x[i], y_left[j][:, i], f"{y_left_colors[j]}o", alpha=i / n_points, label=label)
    plt.legend(loc="upper left")
    plt.xlabel(x_label)
    plt.ylabel("Force (normalized)")

    if y_right is not None:
        # On the right axis, plot the muscle length
        plt.twinx()
        for i in range(n_points):
            label = None
            if i == n_points - 1:
                label = y_right_label
            plt.plot(x[i], y_right[i], f"{y_right_color}o", alpha=i / n_points, label=label)
        plt.legend(loc="upper right")
        plt.ylabel(y_right_label)


def plot_muscle_force_coefficients_surface(
    models: Model | list[Model],
    q: np.ndarray,
    qdot: np.ndarray,
    emg: np.ndarray,
    muscle_index: int | range | slice | None,
    dof_index: int | range | slice | None,
):
    if not isinstance(models, (list, tuple)):
        models = [models]

    # Draw the surface of the force-length-velocity relationship for each model
    fig = plt.figure("Force-Length-Velocity relationship")
    for model in models:
        length, velocity = model.muscles_kinematics(q, qdot)

        x, y = np.meshgrid(q[dof_index, :], qdot[dof_index, :])
        z = np.ndarray((velocity.shape[1], length.shape[1]))

        for i in range(length.shape[1]):
            q_rep = np.repeat(q[:, i : i + 1], qdot.shape[1], axis=1)
            flce, fvce = model.muscle_force_coefficients(emg, q_rep, qdot, muscle_index)
            z[:, i] = flce * fvce

        axis_id = 100 + len(models) * 10 + models.index(model) + 1
        ax = fig.add_subplot(axis_id, projection="3d")
        ax.plot_surface(x, y, z, cmap="viridis")
        ax.set_xlabel("Length")
        ax.set_ylabel("Velocity")
        ax.set_zlabel("Force")
        ax.set_title("Thelen")


def plot_movement(
    t: np.ndarray,
    model: Model,
    q: np.ndarray = None,
    qdot: np.ndarray = None,
    tau: np.ndarray = None,
    emg: np.ndarray = None,
):
    plt.figure()
    n_subplots = sum(x is not None for x in [q, qdot, tau, emg])
    subplot_index = 1

    if q is not None:
        plt.subplot(n_subplots, 1, subplot_index)
        plt.plot(t, q.T)
        plt.title("Position")
        plt.xlabel("Time")
        plt.ylabel("Q")
        subplot_index += 1

    if qdot is not None:
        plt.subplot(n_subplots, 1, subplot_index)
        plt.plot(t, qdot.T)
        plt.title("Velocity")
        plt.xlabel("Time")
        plt.ylabel("Qdot")
        subplot_index += 1

    if tau is not None:
        plt.subplot(n_subplots, 1, subplot_index)
        plt.step(t[[0, -1]], tau[np.newaxis, :][[0, 0], :])
        plt.title("Torque")
        plt.xlabel("Time")
        plt.ylabel("Tau")
        subplot_index += 1

    if emg is not None:
        plt.subplot(n_subplots, 1, subplot_index)
        plt.step(t[[0, -1]], emg[np.newaxis, :][[0, 0], :])
        plt.title("EMG")
        plt.xlabel("Time")
        plt.ylabel("EMG")


def plot_com(
    t: np.ndarray,
    model: Model,
    q: np.ndarray = None,
    qdot: np.ndarray = None,
    tau: np.ndarray = None,
    emg: np.ndarray = None,
):
    plt.figure()
    n_subplots = sum(x is not None for x in [q, qdot])
    subplot_index = 1

    if q is not None:
        plt.subplot(n_subplots, 1, subplot_index)
        plt.plot(t, model.center_of_mass(q))
        plt.title("Center of mass")
        plt.xlabel("Time")
        plt.ylabel("CoM")
        subplot_index += 1

    if qdot is not None:
        plt.subplot(2, 1, 2)
        plt.plot(t, model.center_of_mass_velocity(q, qdot))
        plt.title("Center of mass velocity")
        plt.xlabel("Time")
        plt.ylabel("CoMdot")


def show():
    plt.show()
