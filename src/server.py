import pyautogui as pg
from PIL import Image
import numpy as np
import time
from slot import *
from action import *

import torch
from torchvision.transforms.functional import pil_to_tensor

SAP_PRIVATE_GAME_NAME = "dogparkies"
CF = 0.8

class Battle(Enum):
    WIN = 0
    DRAW = 1
    LOSS = 2
    RUN_WIN = 3
    RUN_LOSS = 4
    ONGOING = 5

class Role(Enum):
    PLAYER = 0
    HOST = 1

class SAPServer:
    res = {
        "arena": "resources/arena.png",
        "icon": "resources/window_icon.png",
        "victory": "resources/victory.png",
        "defeat": "resources/defeat.png",
        "draw": "resources/draw.png",
        "zero_gold": "resources/zero_gold.png",
        "one_gold": "resources/one_gold.png",
        "two_gold": "resources/two_gold.png",
        "ten_gold": "resources/ten_gold.png",
        "gameover": "resources/gameover.png",
        "arena_won": "resources/arena_won.png",
        "pause_button": "resources/pause_button.png",
        "ff_button": "resources/ff_button.png",
        "roll": "resources/roll.png"
    }

    def __init__(self, role):
        self.role = role
        #icon_loc = pg.locateOnScreen(Image.open(self.res["icon"]), confidence=CF)        
        #if (icon_loc is None):
        #    print("StateServer failed to find SAP window.")
        #    exit(1)

        self.window_loc = (SAP_WINDOW_L, SAP_WINDOW_T, 
                           SAP_WINDOW_W, SAP_WINDOW_H)

    def get_full_state(self):
        return pg.screenshot(region=self.window_loc)

    def start_run(self):
        #self.join_private_match()
        #self.start_private_match()
        pg.moveTo(ARENA_LOC, duration=0.2)
        pg.doubleClick()
        time.sleep(1)

    def start_battle(self, state):
        self.apply(Action.A68)
        self.press_button(CONFIRM_LOC)

    def press_button(self, loc):
        pg.moveTo(loc, duration=0.2)
        pg.click(clicks=2, interval=0.2)
        time.sleep(1)

    def click_center(self):
        pg.moveTo(ARENA_LOC)
        pg.click()

    def click_top(self):
        pg.moveTo(HOVER_LOC)
        pg.click()

    def battle_ready(self, state):
        pause_search = pg.locate(Image.open(self.res["ff_button"]), state, confidence=CF)
        if (pause_search):
            return True
        return False

    def battle_status(self, state):
        victory_search = pg.locate(Image.open(self.res["victory"]), state, confidence=CF)
        if (victory_search):
            print("Victory")
            return Battle.WIN

        defeat_search = pg.locate(Image.open(self.res["defeat"]), state, confidence=CF)
        if (defeat_search):
            print("Defeat")
            return Battle.LOSS

        draw_search = pg.locate(Image.open(self.res["draw"]), state, confidence=CF)
        if (draw_search):
            print("Draw")
            return Battle.DRAW

        arena_won_search = pg.locate(Image.open(self.res["arena_won"]), state, confidence=CF)
        if (arena_won_search):
            print("Run Won")
            return Battle.RUN_WIN

        arena_lost_search = pg.locate(Image.open(self.res["gameover"]), state, confidence=0.6)
        if (arena_lost_search):
            print("Run Lost")
            return Battle.RUN_LOSS

        return Battle.ONGOING

    def run_complete(self, state):
        arena_search = pg.locate(Image.open(self.res["arena"]), state, confidence=CF)
        if (arena_search):
            return True
        return False

    def buy_complete(self, state):
        gold_search = pg.locate(Image.open(self.res["zero_gold"]), state, confidence=CF)
        if (gold_search):
            self.start_battle()
            return True
        return False

    def shop_ready(self, state):
        sign_search = pg.locate(Image.open(self.res["roll"]), state, confidence=CF)
        if (sign_search):
            return True
        return False

    def low_gold(self, state):
        ten_search = pg.locate(Image.open(self.res["ten_gold"]), state, confidence=CF)
        if (ten_search):
            return False

        zero_search = pg.locate(Image.open(self.res["zero_gold"]), state, confidence=CF)
        if (zero_search):
            print("Zero gold")
            return True

        one_search = pg.locate(Image.open(self.res["one_gold"]), state, confidence=0.9)
        if (one_search):
            print("One gold")
            return True

        two_search = pg.locate(Image.open(self.res["two_gold"]), state, confidence=CF)
        if (two_search):
            print("Two gold")
            return True
        return False

    def zero_gold(self, state):
        zero_search = pg.locate(Image.open(self.res["zero_gold"]), state, confidence=CF)
        if (zero_search):
            return True
        return False

    def get_appropriate_mask(self, state, turn, step):
        mask = SAP_ACTION_NO_MASK.clone()

        if (self.low_gold(state)):
            print("Masking buy actions.")
            return SAP_ACTION_ALL_BUY_MASK.clone()
        else:
            mask[:,-1] = 0

        if (turn < 3):
            mask = SAP_ACTION_TURN_ONE_MASK * mask
        elif (turn < 5):
            mask = SAP_ACTION_TURN_THREE_MASK * mask
        elif (turn < 9):
            mask = SAP_ACTION_TURN_FIVE_MASK * mask

        print(mask)
        return mask

    def reward_default(self, battle_status):
        ''' The default reward is to deliver the battle status enum value '''
        if (battle_status is Battle.WIN):
            return 1
        if (battle_status is Battle.DRAW):
            return 1
        if (battle_status is Battle.LOSS):
            return -1
        if (battle_status is Battle.GAMEOVER):
            return 0

    def reward_duration(self, battle_status, duration):
        print("Duration: ", duration)
        base = 0
        if (battle_status is Battle.WIN or battle_status is Battle.RUN_WIN or battle_status is Battle.DRAW):
            base = 1
        elif (battle_status is Battle.LOSS or battle_status is Battle.RUN_LOSS):
            base = -1
        return base * (20.0 - duration)

    def apply(self, action):
        """ Apply the given action in SAP """
        f, args = SAP_ACTION_FUNC[action]
        f(args)
        self.hover()

    def hover(self):
        pg.moveTo(HOVER_LOC)
        pg.click()

    def join_private_match(self):
        self.press_button(VERSUS_LOC)
        if (self.role is Role.PLAYER):
            time.sleep(5)

        if (self.role is Role.HOST): 
           self.press_button(CREATE_PRIVATE_LOC)
        else:
           self.press_button(JOIN_PRIVATE_LOC)
        
        self.press_button(ENTER_NAME_LOC)
        pg.write(SAP_PRIVATE_GAME_NAME)
        self.press_button(CONFIRM_PRIVATE_LOC)

    def start_private_match(self):
        if (self.role is Role.HOST):
            while(self.shop_ready(self.get_full_state()) is False):
                print("Waiting for players")
                time.sleep(2)
                self.press_button(START_GAME_LOC)
        else:
            while(self.shop_ready(self.get_full_state()) is False):
                print("Waiting for game to start")
                time.sleep(5)
