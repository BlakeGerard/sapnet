# SAPNet requirements
from slot import *
from action import *
from mouse import *
from image import GrimImager

# Image processing
from PIL import Image
from PIL import ImageGrab
import cv2 as cv

# Input and timing
import time
import os

# Pytorch
import torch
import numpy as np

SAP_PRIVATE_GAME_NAME = "dogparkies"

CV_METHOD = cv.TM_CCOEFF_NORMED
CV_CONF = 0.99
RESOLUTION = "240p"

res = {
    "logout"    : f"resources/{RESOLUTION}/logout.png",
    "victory"   : f"resources/{RESOLUTION}/victory.png",
    "defeat"    : f"resources/{RESOLUTION}/defeat.png",
    "draw"      : f"resources/{RESOLUTION}/draw.png",
    "gameover"  : f"resources/{RESOLUTION}/gameover.png",
    "pause"     : f"resources/{RESOLUTION}/pause.png",
    "roll"      : f"resources/{RESOLUTION}/roll.png",
    "zero_gold" : f"resources/{RESOLUTION}/zero_gold.png",
    "one_gold"  : f"resources/{RESOLUTION}/one_gold.png",
    "two_gold"  : f"resources/{RESOLUTION}/two_gold.png",
    "ten_gold"  : f"resources/{RESOLUTION}/ten_gold.png",
}


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
    def __init__(self, role):
        self.role = role
        self.window_loc = (SAP_WINDOW_L, SAP_WINDOW_T, SAP_WINDOW_W, SAP_WINDOW_H)
        self.mouse = Mouse("/dev/input/event13", ARENA_LOC)
        self.viewer = GrimImager("DP-2")

    # Apply an action
    def apply(self, action):
        self.mouse.click()
        f, args = SAP_ACTION_FUNC[action]
        f(self.mouse, args)
        action_hover(self.mouse, args)

    # Capture the entire game state
    def get_full_state(self):
        pil_image = self.viewer.capture_area(self.window_loc)
        return np.asarray(pil_image)

    def open_image_as_np_array(self, path):
        pil_image = Image.open(path)
        return np.asarray(pil_image)

    # Check if game element is present in the game state
    def locate(self, needle, haystack, confidence):
        res = cv.matchTemplate(haystack, needle, CV_METHOD)
        if np.amax(res) > CV_CONF:
            return True
        return False

    # Check if the battle has started by looking for the fast forward button
    def battle_ready(self, state):
        pause_search = self.locate(
            self.open_image_as_np_array(res["pause"]), state, confidence=CV_CONF
        )
        if pause_search:
            return True
        return False

    # Check battle status by looking for battle result screen splash
    def battle_status(self, state):
        victory_search = self.locate(
            self.open_image_as_np_array(res["victory"]), state, confidence=CV_CONF
        )
        if victory_search:
            print("Victory")
            return Battle.WIN

        defeat_search = self.locate(self.open_image_as_np_array(res["defeat"]), state, confidence=CV_CONF)
        if defeat_search:
            print("Defeat")
            return Battle.LOSS

        draw_search = self.locate(self.open_image_as_np_array(res["draw"]), state, confidence=CV_CONF)
        if draw_search:
            print("Draw")
            return Battle.DRAW

        arena_lost_search = self.locate(
            self.open_image_as_np_array(res["gameover"]), state, confidence=CV_CONF
        )
        if arena_lost_search:
            print("Run Lost")
            return Battle.RUN_LOSS

        return Battle.ONGOING

    # Check if the entire run is complete by looking for the arena button
    def run_complete(self, state):
        arena_search = self.locate(self.open_image_as_np_array(res["logout"]), state, confidence=CV_CONF)
        if arena_search:
            return True
        return False

    # Check that the shop is ready by looking for the roll button
    def shop_ready(self, state):
        sign_search = self.locate(self.open_image_as_np_array(res["roll"]), state, confidence=CV_CONF)
        if sign_search:
            return True
        return False

    # Check if we are below three gold
    def low_gold(self, state):
        ten_search = self.locate(self.open_image_as_np_array(res["ten_gold"]), state, confidence=CV_CONF)
        if ten_search:
            return False

        zero_search = self.locate(
            self.open_image_as_np_array(res["zero_gold"]), state, confidence=CV_CONF
        )
        if zero_search:
            print("Zero gold")
            return True

        one_search = self.locate(self.open_image_as_np_array(res["one_gold"]), state, confidence=0.9)
        if one_search:
            print("One gold")
            return True

        two_search = self.locate(self.open_image_as_np_array(res["two_gold"]), state, confidence=CV_CONF)
        if two_search:
            print("Two gold")
            return True
        return False

    # Check if we are totally out of gold
    def zero_gold(self, state):
        zero_search = self.locate(
            self.open_image_as_np_array(res["zero_gold"]), state, confidence=CV_CONF
        )
        if zero_search:
            return True
        return False

    # Mask decisions based on turn and gold state
    def get_appropriate_mask(self, state, turn, step):
        mask = SAP_ACTION_NO_MASK.clone()

        if self.low_gold(state):
            print("Masking buy actions.")
            return SAP_ACTION_ALL_BUY_MASK.clone()
        else:
            mask[:, -1] = 0

        if turn < 3:
            mask = SAP_ACTION_TURN_ONE_MASK * mask
        elif turn < 5:
            mask = SAP_ACTION_TURN_THREE_MASK * mask
        elif turn < 9:
            mask = SAP_ACTION_TURN_FIVE_MASK * mask
        return mask

    # Default integer reward function
    def reward_default(self, battle_status):
        if battle_status is Battle.WIN:
            return 1
        if battle_status is Battle.DRAW:
            return 1
        if battle_status is Battle.LOSS:
            return -1
        if battle_status is Battle.RUN_LOSS:
            return 0

    def reward_duration(self, battle_status, duration):
        base = 0
        if (
            battle_status is Battle.WIN
            or battle_status is Battle.RUN_WIN
            or battle_status is Battle.DRAW
        ):
            base = 1
        elif battle_status is Battle.LOSS or battle_status is Battle.RUN_LOSS:
            base = -1
        return base * (20.0 - duration)

    # Game interaction to begin arena or private match
    def begin_arena_run(self):
        print("Move the cursor to (0,0) to begin")
        time.sleep(5)
        self.mouse.move_and_click(ARENA_LOC)
        time.sleep(3)

    def begin_private_match(self):
        self.join_private_match()
        self.start_private_match()

    def start_battle(self, state):
        self.apply(Action.A68)
        self.mouse.move_and_click(CONFIRM_LOC)

    def join_private_match(self):
        self.press_button(VERSUS_LOC)
        if self.role is Role.PLAYER:
            time.sleep(5)

        if self.role is Role.HOST:
            self.press_button(CREATE_PRIVATE_LOC)
        else:
            self.press_button(JOIN_PRIVATE_LOC)

        self.press_button(ENTER_NAME_LOC)
        pg.write(SAP_PRIVATE_GAME_NAME)
        self.press_button(CONFIRM_PRIVATE_LOC)

    def start_private_match(self):
        if self.role is Role.HOST:
            while self.shop_ready(self.get_full_state()) is False:
                print("Waiting for players")
                time.sleep(5)
                self.press_button(START_GAME_LOC)
        else:
            while self.shop_ready(self.get_full_state()) is False:
                print("Waiting for game to start")
                time.sleep(5)
