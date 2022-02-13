from state import StateServer
from action import *

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch

srv = StateServer()
#srv.start()
state = srv.get_state()

# Buy shop pet at Slot.B0 and place it in team Slot.T0
srv.apply(Action.A10)
srv.apply(Action.A40)