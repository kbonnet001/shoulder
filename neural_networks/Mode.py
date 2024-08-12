from enum import Enum

class Mode(Enum):
    """ Enumeration for different modes of operation.

    Attributes:
    - MUSCLE: Mode for muscle (lmt) only.
    - DLMT_DQ: Mode for muscle length jacobian (dlmt_dq) only.
    - MUSCLE_DLMT_DQ: Mode for both muscle (lmt) and muscle length jacobian (dlmt_dq).
    - TORQUE: Mode for torque only.
    - TORQUE_MUS_DLMT_DQ: Mode for torque, muscle (lmt), and muscle length jacobian (dlmt_dq).
    """

    MUSCLE = 1 
    DLMT_DQ = 2 
    MUSCLE_DLMT_DQ = 3 
    TORQUE = 4 
    TORQUE_MUS_DLMT_DQ = 5 