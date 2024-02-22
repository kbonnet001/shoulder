from enum import Enum, auto


class MuscleCoefficientPlots(Enum):
    NONE = auto()
    TIME = auto()
    MUSCLE_PARAMETERS = auto()
    KINEMATICS = auto()


class MuscleSurfacePlots(Enum):
    NONE = auto()
    COEFFICIENTS = auto()
    FORCE = auto()
