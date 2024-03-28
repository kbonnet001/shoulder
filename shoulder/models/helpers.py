import numpy as np
from casadi import SX, MX, DM, vertcat


type Vector = np.ndarray | SX | MX | DM
type Scalar = float | int | SX | MX | DM


def parse_muscle_index(muscle_index: range | slice | int | None, n_muscles: int) -> slice:
    if muscle_index is None:
        return slice(0, n_muscles)
    elif isinstance(muscle_index, int):
        return slice(muscle_index, muscle_index + 1)
    elif isinstance(muscle_index, range):
        return slice(muscle_index.start, muscle_index.stop)
    else:
        raise ValueError("muscle_index must be an int, a range or a slice")


def concatenate(*args: Vector) -> Vector:
    if isinstance(args[0], (SX, MX, DM)):
        return vertcat(*args)
    elif isinstance(args[0], np.ndarray):
        return np.concatenate(args, axis=0)
    else:
        raise ValueError("Unsupported type for concatenation")
