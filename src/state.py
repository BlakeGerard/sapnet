import pyautogui as pg
import numpy as np
import time
from slot import *
from action import *

import torch
import torchvision.transforms as T

class StateServer:
    res = {
        "arena" : "resources/arena.png",
        "icon": "resources/window_icon.png"
    }

    def __init__(self):
        icon_loc = pg.locateOnScreen(self.res["icon"], confidence=0.5)
        
        if (icon_loc is None):
            print("StateServer failed to find SAP window.")
            exit(1)

        self.window_loc = (SAP_WINDOW_L, SAP_WINDOW_T, 
                           SAP_WINDOW_W, SAP_WINDOW_H)

    def start(self):
        self.arena_loc = pg.locateCenterOnScreen(self.res["arena"], confidence=0.5)
        if (self.arena_loc is None):
            print("StateServer could not find Arena Mode button.")
            exit(1)
        pg.click(self.arena_loc, clicks=2, duration=0.5)
        time.sleep(10)

    def get_state(self):
        state = pg.screenshot(region=self.window_loc)
        state = np.ascontiguousarray(state)
        state = torch.from_numpy(state)
        state = state.unsqueeze(0)
        return state

    def apply(self, action):
        """ Apply the given action in SAP """
        f, args = SAP_ACTION_FUNC[action]
        f(args)