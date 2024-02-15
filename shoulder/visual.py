import bioviz
import numpy as np
from matplotlib import pyplot as plt

from .model import Model


def animate(q: np.ndarray, model: Model):
    viz = bioviz.Viz(loaded_model=model.model)
    viz.load_movement(q)
    viz.set_camera_roll(np.pi / 2)
    viz.exec()


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
