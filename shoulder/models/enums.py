from enum import Enum


class ControlsTypes(Enum):
    EMG = 0
    TORQUE = 1


class IntegrationMethods(Enum):
    RK4 = "RK4"
    RK45 = "RK45"


class MuscleParameter(Enum):
    OPTIMAL_LENGTH = "optimal_length"
    TENDON_SLACK_LENGTH = "tendon_slack_length"
    PENNATION_ANGLE = "pennation_angle"
