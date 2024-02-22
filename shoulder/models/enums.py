from enum import Enum


class ControlsTypes(Enum):
    EMG = 0
    TORQUE = 1


class IntegrationMethods(Enum):
    RK4 = "RK4"
    RK45 = "RK45"
