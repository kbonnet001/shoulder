from enum import Enum

class Mode(Enum):
    MUSCLE = 1 # muscle only
    DLMT_DQ = 2 # lever arm dlmt_dq only
    MUSCLE_DLMT_DQ = 3 # muscle and lever arm
    TORQUE = 4 # torque only
    TORQUE_MUS_DLMT_DQ = 5 # torque, muscle and dlmt_dq