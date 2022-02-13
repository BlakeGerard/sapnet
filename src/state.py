import pyautogui as pg
import numpy as np
import slot

SAP_WINDOW_W = 960
SAP_WINDOW_H = 629

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

        # Set pixels from left, pixels from top, width, height
        self.window_loc = (icon_loc.left, icon_loc.top * 16, 
                           SAP_WINDOW_W, SAP_WINDOW_H)

    def start(self):
        self.arena_loc = pg.locateCenterOnScreen(self.res["arena"], confidence=0.5)
        if (self.arena_loc is None):
            print("StateServer could not find Arena Mode button.")
            exit(1)
        pg.click(self.arena_loc, clicks=2, duration=0.5)
        sleep(10)

    def get_state(self):
        pil = pg.screenshot(region=self.window_loc)
        return np.asarray(pil)

    def action_swap(self, src, dst):
        assert(src in TEAM_SLOTS and dst in TEAM_SLOTS)

    def action_buy(self, src, dst):
        assert(src in BUY_SLOTS and dst in TEAM_SLOTS)

    def action_freeze(self, dst):
        assert(dst in BUY_SLOTS)

    def action_roll(self):
        pass
        
    def action_end(self):
        pass