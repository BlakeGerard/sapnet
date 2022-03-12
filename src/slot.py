from enum import Enum 

SAP_WINDOW_L = 0
SAP_WINDOW_T = 30
SAP_WINDOW_W = 480
SAP_WINDOW_H = 386   # 416 - 30

class Slot(Enum):
    T0 = 0
    T1 = 1
    T2 = 2
    T3 = 3
    T4 = 4
    B0 = 5
    B1 = 6
    B2 = 7
    B3 = 8
    B4 = 9
    B5 = 10
    B6 = 11

TEAM_ROW = 0.41 * SAP_WINDOW_H
BUY_ROW = 0.61 * SAP_WINDOW_H
COL_0 = 0.24 * SAP_WINDOW_W
COL_1 = 0.33 * SAP_WINDOW_W
COL_2 = 0.41 * SAP_WINDOW_W
COL_3 = 0.50 * SAP_WINDOW_W
COL_4 = 0.58 * SAP_WINDOW_W
COL_5 = 0.65 * SAP_WINDOW_W
COL_6 = 0.74 * SAP_WINDOW_W

SLOT_LOC = {
    Slot.T0 : (COL_0 + SAP_WINDOW_L, TEAM_ROW + SAP_WINDOW_T),
    Slot.T1 : (COL_1 + SAP_WINDOW_L, TEAM_ROW + SAP_WINDOW_T),
    Slot.T2 : (COL_2 + SAP_WINDOW_L, TEAM_ROW + SAP_WINDOW_T),
    Slot.T3 : (COL_3 + SAP_WINDOW_L, TEAM_ROW + SAP_WINDOW_T),
    Slot.T4 : (COL_4 + SAP_WINDOW_L, TEAM_ROW + SAP_WINDOW_T),
    Slot.B0 : (COL_0 + SAP_WINDOW_L, BUY_ROW + SAP_WINDOW_T),
    Slot.B1 : (COL_1 + SAP_WINDOW_L, BUY_ROW + SAP_WINDOW_T),
    Slot.B2 : (COL_2 + SAP_WINDOW_L, BUY_ROW + SAP_WINDOW_T),
    Slot.B3 : (COL_3 + SAP_WINDOW_L, BUY_ROW + SAP_WINDOW_T),
    Slot.B4 : (COL_4 + SAP_WINDOW_L, BUY_ROW + SAP_WINDOW_T),
    Slot.B5 : (COL_5 + SAP_WINDOW_L, BUY_ROW + SAP_WINDOW_T),
    Slot.B6 : (COL_6 + SAP_WINDOW_L, BUY_ROW + SAP_WINDOW_T)
}

HOVER_LOC           = (0.5 * SAP_WINDOW_W + SAP_WINDOW_L, 0.1 * SAP_WINDOW_H + SAP_WINDOW_T)
ARENA_LOC           = (0.5 * SAP_WINDOW_W + SAP_WINDOW_L, 0.5 * SAP_WINDOW_H + SAP_WINDOW_T)
CONFIRM_LOC         = (0.65 * SAP_WINDOW_W + SAP_WINDOW_L, 0.62 * SAP_WINDOW_H + SAP_WINDOW_T)
ROLL_LOC            = (0.1 * SAP_WINDOW_W + SAP_WINDOW_L, 0.9 * SAP_WINDOW_H + SAP_WINDOW_T)
SELL_LOC            = (0.5 * SAP_WINDOW_W + SAP_WINDOW_L, 0.9 * SAP_WINDOW_H + SAP_WINDOW_T)
FREEZE_LOC          = (0.5 * SAP_WINDOW_W + SAP_WINDOW_L, 0.9 * SAP_WINDOW_H + SAP_WINDOW_T)
END_LOC             = (0.8 * SAP_WINDOW_W + SAP_WINDOW_L, 0.9 * SAP_WINDOW_H + SAP_WINDOW_T)
VERSUS_LOC          = (0.5 * SAP_WINDOW_W + SAP_WINDOW_L, 0.75 * SAP_WINDOW_H + SAP_WINDOW_T)
CREATE_PRIVATE_LOC  = (0.5 * SAP_WINDOW_W + SAP_WINDOW_L, 0.6 * SAP_WINDOW_H + SAP_WINDOW_T)
JOIN_PRIVATE_LOC    = (0.5 * SAP_WINDOW_W + SAP_WINDOW_L, 0.47 * SAP_WINDOW_H + SAP_WINDOW_T)
ENTER_NAME_LOC      = (0.5 * SAP_WINDOW_W + SAP_WINDOW_L, 0.4 * SAP_WINDOW_H + SAP_WINDOW_T)
CONFIRM_PRIVATE_LOC = (0.5 * SAP_WINDOW_W + SAP_WINDOW_L, 0.53 * SAP_WINDOW_H + SAP_WINDOW_T)
START_GAME_LOC      = (0.37 * SAP_WINDOW_W + SAP_WINDOW_L, 0.3 * SAP_WINDOW_H + SAP_WINDOW_T)

'''
SLOT_LOC = {
    Slot.T0 : (240, 240),
    Slot.T1 : (320, 240),
    Slot.T2 : (400, 240),
    Slot.T3 : (480, 240),
    Slot.T4 : (560, 240),
    Slot.B0 : (240, 400),
    Slot.B1 : (320, 400),
    Slot.B2 : (400, 400),
    Slot.B3 : (480, 400),
    Slot.B4 : (560, 400),
    Slot.B5 : (640, 400),
    Slot.B6 : (720, 400)
}
'''

'''
HOVER_LOC = (480, 50)
ARENA_LOC = (480, 300)
CONFIRM_LOC = (630, 400)
ROLL_LOC = (120, 580)
SELL_LOC = (440, 580)
FREEZE_LOC = (440, 580)
END_LOC = (790, 580)
VERSUS_LOC = (483, 425)
CREATE_PRIVATE_LOC = (472, 377)
JOIN_PRIVATE_LOC = (472, 285)
ENTER_NAME_LOC = (482, 242)
CONFIRM_PRIVATE_LOC = (474, 332)
START_GAME_LOC = (338, 199)
'''
