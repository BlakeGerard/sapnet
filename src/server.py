import pyautogui as pg
from PIL import Image
import numpy as np
import time
from slot import *
from action import *

import torch
from torchvision.transforms.functional import pil_to_tensor

class Battle(Enum):
    WIN = 0
    LOSS = 1
    DRAW = 2
    ONGOING = 3
    GAMEOVER = 4

class SAPServer:
    res = {
        "arena" : "resources/arena.png",
        "icon": "resources/window_icon.png",
        "victory": "resources/victory.png",
        "defeat": "resources/defeat.png",
        "draw": "resources/draw.png",
        "confirm": "resources/confirm.png",
        "zero_gold": "resources/zero_gold.png",
        "one_gold": "resources/one_gold.png",
        "two_gold": "resources/two_gold.png",
        "ten_gold": "resources/ten_gold.png",
        "gameover": "resources/gameover.png",
        "slot": "resources/slot.png"
    }

    def __init__(self):
        icon_loc = pg.locateOnScreen(Image.open(self.res["icon"]), confidence=0.5)
        
        if (icon_loc is None):
            print("StateServer failed to find SAP window.")
            exit(1)

        self.window_loc = (SAP_WINDOW_L, SAP_WINDOW_T, 
                           SAP_WINDOW_W, SAP_WINDOW_H)

    def get_state(self):
        return pg.screenshot(region=self.window_loc)

    def start_run(self):
        pg.moveTo(ARENA_LOC, duration=0.5)
        pg.doubleClick()
        time.sleep(1)

    def start_battle(self, state):
        self.apply(Action.A58)
        self.press_button(CONFIRM_LOC)

    def press_button(self, loc):
        pg.moveTo(loc)
        pg.click()

    def click_center(self):
        pg.moveTo(ARENA_LOC)
        pg.click()

    def battle_status(self, state):
        victory_search = pg.locate(Image.open(self.res["victory"]), state, confidence=0.5)
        if (victory_search):
            return Battle.WIN

        defeat_search = pg.locate(Image.open(self.res["defeat"]), state, confidence=0.5)
        if (defeat_search):
            return Battle.LOSS

        draw_search = pg.locate(Image.open(self.res["draw"]), state, confidence=0.5)
        if (draw_search):
            return Battle.DRAW

        gameover_search = pg.locate(Image.open(self.res["gameover"]), state, confidence=0.5)
        if (gameover_search):
            return Battle.GAMEOVER

        return Battle.ONGOING

    def run_complete(self, state):
        arena_search = pg.locate(Image.open(self.res["arena"]), state, confidence=0.5)
        if (arena_search):
            return True
        return False

    def buy_complete(self, state):
        gold_search = pg.locate(Image.open(self.res["zero_gold"]), state, confidence=0.5)
        if (gold_search):
            self.start_battle()
            return True
        return False

    def low_gold(self, state):
        zero_search = pg.locate(Image.open(self.res["zero_gold"]), state, confidence=0.95)
        if (zero_search):
            return True

        one_search = pg.locate(Image.open(self.res["one_gold"]), state, confidence=0.95)
        if (one_search):
            return True

        two_search = pg.locate(Image.open(self.res["two_gold"]), state, confidence=0.95)
        if (two_search):
            return True
        return False

    def zero_gold(self, state):
        zero_search = pg.locate(Image.open(self.res["zero_gold"]), state, confidence=0.95)
        if (zero_search):
            return True
        return False

    def get_appropriate_mask(self, state, turn):

        # If we have gold < 3, mask off all buy actions
        if (self.low_gold(state)):
            print("Gold is low. Masking buy actions.")
            return SAP_ACTION_LOW_GOLD_MASK

        #t0_search = pg.locate(Image.open(self.res["slot"]), state, region = T0_REGION, confidence=0.1)
        #t1_search = pg.locate(Image.open(self.res["slot"]), state, region = T1_REGION, confidence=0.1)
        #t2_search = pg.locate(Image.open(self.res["slot"]), state, region = T2_REGION, confidence=0.1)
        #t3_search = pg.locate(Image.open(self.res["slot"]), state, region = T3_REGION, confidence=0.1)
        #t4_search = pg.locate(Image.open(self.res["slot"]), state, region = T4_REGION, confidence=0.1)
        #if (t0_search is None and
        #    t1_search is None and
        #    t2_search is None and
        #    t3_search is None and
        #    t4_search is None):
        #    print("Team is full. Masking pet buy actions.")
        #    return SAP_ACTION_NO_PET_BUY_MASK

        # Otherwise, mask off specific buy actions by turn
        if (turn < 3):
            return SAP_ACTION_TURN_ONE_MASK
        elif (turn < 5):
            return SAP_ACTION_TURN_THREE_MASK
        elif (turn < 9):
            return SAP_ACTION_TURN_FIVE_MASK

    def reward(self, battle_status):
        assert(battle_status is not Battle.ONGOING)
        if (battle_status is Battle.WIN or battle_status is Battle.DRAW):
            return 1
        if (battle_status is Battle.LOSS):
            return -1
        return 0

    def apply(self, action):
        """ Apply the given action in SAP """
        f, args = SAP_ACTION_FUNC[action]
        f(args)
        time.sleep(0.5)