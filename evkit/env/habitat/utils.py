import torch
from habitat.sims.habitat_simulator import SimulatorActions

try:
    from habitat.sims.habitat_simulator import SIM_NAME_TO_ACTION
except:
    pass


# TODO these are action values. Make sure to add the word "action" into the name
FORWARD_VALUE = SimulatorActions.FORWARD.value
FORWARD_VALUE = FORWARD_VALUE if isinstance(FORWARD_VALUE, int) else SIM_NAME_TO_ACTION[FORWARD_VALUE]

STOP_VALUE = SimulatorActions.STOP.value
STOP_VALUE = STOP_VALUE if isinstance(STOP_VALUE, int) else SIM_NAME_TO_ACTION[STOP_VALUE]

LEFT_VALUE = SimulatorActions.LEFT.value
LEFT_VALUE = LEFT_VALUE if isinstance(LEFT_VALUE, int) else SIM_NAME_TO_ACTION[LEFT_VALUE]

RIGHT_VALUE = SimulatorActions.RIGHT.value
RIGHT_VALUE = RIGHT_VALUE if isinstance(RIGHT_VALUE, int) else SIM_NAME_TO_ACTION[RIGHT_VALUE]


TAKEOVER1 = [LEFT_VALUE] * 4 + [FORWARD_VALUE] * 4
TAKEOVER2 = [RIGHT_VALUE] * 4 + [FORWARD_VALUE] * 4
TAKEOVER3 = [LEFT_VALUE] * 6 + [FORWARD_VALUE] * 2
TAKEOVER4 = [RIGHT_VALUE] * 6 + [FORWARD_VALUE] * 2
# TAKEOVER5 = [LEFT_VALUE] * 8  # rotation only seems not to break out of bad behavior
# TAKEOVER6 = [RIGHT_VALUE] * 8
TAKEOVER_ACTION_SEQUENCES = [TAKEOVER1, TAKEOVER2, TAKEOVER3, TAKEOVER4]
TAKEOVER_ACTION_SEQUENCES = [torch.Tensor(t).long() for t in TAKEOVER_ACTION_SEQUENCES]

DEFAULT_TAKEOVER_ACTIONS = torch.Tensor([LEFT_VALUE, LEFT_VALUE, LEFT_VALUE, LEFT_VALUE, FORWARD_VALUE, FORWARD_VALUE]).long()
NON_STOP_VALUES = torch.Tensor([FORWARD_VALUE, LEFT_VALUE, RIGHT_VALUE]).long()
