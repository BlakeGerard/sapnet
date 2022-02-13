from enum import Enum
from slot import *
import pyautogui as pg

def action_move(args):
    """ Move a team pet from slot args[0] to slot args[1] """
    assert(len(args) == 2)
    pg.moveTo(SLOT_LOC[args[0]], duration=0.2)
    pg.click()
    pg.dragTo(SLOT_LOC[args[1]], duration=0.2)

def action_buy(args):
    """ Buy the shop pet at args[0] and place in team position args[1] """
    assert(len(args) == 2)
    pg.moveTo(SLOT_LOC[args[0]], duration=0.2)
    pg.click()
    pg.dragTo(SLOT_LOC[args[1]], duration=0.2)

def action_sell(args):
    """ Sell the team pet at args """
    assert(args is not None)
    pg.moveTo(SLOT_LOC[args], duration=0.2)
    pg.doubleClick()
    pg.moveTo(SELL_LOC, duration=0.2)
    pg.click()

def action_freeze(args):
    """ Freeze the shop pet at args """
    assert(args is not None)
    pg.displayMousePosition()
    pg.moveTo(SLOT_LOC[args], duration=0.2)
    pg.doubleClick()
    pg.moveTo(FREEZE_LOC, duration=0.2)
    pg.click()

def action_roll(args):
    """ Roll """
    assert(args is None)
    pg.moveTo(ROLL_LOC, duration=0.2)
    pg.doubleClick()

def action_end(args):
    """ End turn """
    assert(args is None)
    pg.moveTo(END_LOC, duration=0.2)
    pg.click(clicks=3)

class Action(Enum):
    A0 = 0
    A1 = 1
    A2 = 2
    A3 = 3
    A4 = 4
    A5 = 5
    A6 = 6
    A7 = 7
    A8 = 8
    A9 = 9
    A10 = 10
    A11 = 11
    A12 = 12
    A13 = 13
    A14 = 14
    A15 = 15
    A16 = 16
    A17 = 17
    A18 = 18
    A19 = 19
    A20 = 20
    A21 = 21
    A22 = 22
    A23 = 23
    A24 = 24
    A25 = 25
    A26 = 26
    A27 = 27
    A28 = 28
    A29 = 29
    A30 = 30
    A31 = 31
    A32 = 32
    A33 = 33
    A34 = 34
    A35 = 35
    A36 = 36
    A37 = 37
    A38 = 38
    A39 = 39
    A40 = 40
    A41 = 41
    A42 = 42
    A43 = 43
    A44 = 44
    A45 = 45
    A46 = 46
    A47 = 47
    A48 = 48
    A49 = 49
    A50 = 50
    A51 = 51
    A52 = 52
    A53 = 53
    A54 = 54
    A55 = 55
    A56 = 56
    A57 = 57
    A58 = 58

SAP_ACTION_FUNC = {

    # Move friends in the team
    Action.A0  : (action_move, (Slot.T0, Slot.T1)),
    Action.A1  : (action_move, (Slot.T0, Slot.T2)),
    Action.A2  : (action_move, (Slot.T0, Slot.T3)),
    Action.A3  : (action_move, (Slot.T0, Slot.T4)),
    Action.A4  : (action_move, (Slot.T1, Slot.T2)),
    Action.A5  : (action_move, (Slot.T1, Slot.T3)),
    Action.A6  : (action_move, (Slot.T1, Slot.T4)),
    Action.A7  : (action_move, (Slot.T2, Slot.T3)),
    Action.A8  : (action_move, (Slot.T2, Slot.T4)),
    Action.A9  : (action_move, (Slot.T3, Slot.T4)),

    # Buy from a BUY_SLOT and move to a TEAM_SLOT
    Action.A10 : (action_buy, (Slot.B0, Slot.T0)),
    Action.A11 : (action_buy, (Slot.B0, Slot.T1)),
    Action.A12 : (action_buy, (Slot.B0, Slot.T2)),
    Action.A13 : (action_buy, (Slot.B0, Slot.T3)),
    Action.A14 : (action_buy, (Slot.B0, Slot.T4)),
    Action.A15 : (action_buy, (Slot.B1, Slot.T0)),
    Action.A16 : (action_buy, (Slot.B1, Slot.T1)),
    Action.A17 : (action_buy, (Slot.B1, Slot.T2)),
    Action.A18 : (action_buy, (Slot.B1, Slot.T3)),
    Action.A19 : (action_buy, (Slot.B1, Slot.T4)),
    Action.A20 : (action_buy, (Slot.B2, Slot.T0)),
    Action.A21 : (action_buy, (Slot.B2, Slot.T1)),
    Action.A22 : (action_buy, (Slot.B2, Slot.T2)),
    Action.A23 : (action_buy, (Slot.B2, Slot.T3)),
    Action.A24 : (action_buy, (Slot.B2, Slot.T4)),
    Action.A25 : (action_buy, (Slot.B3, Slot.T0)),
    Action.A26 : (action_buy, (Slot.B3, Slot.T1)),
    Action.A27 : (action_buy, (Slot.B3, Slot.T2)),
    Action.A28 : (action_buy, (Slot.B3, Slot.T3)),
    Action.A29 : (action_buy, (Slot.B3, Slot.T4)),
    Action.A30 : (action_buy, (Slot.B4, Slot.T0)),
    Action.A31 : (action_buy, (Slot.B4, Slot.T1)),
    Action.A32 : (action_buy, (Slot.B4, Slot.T2)),
    Action.A33 : (action_buy, (Slot.B4, Slot.T3)),
    Action.A34 : (action_buy, (Slot.B4, Slot.T4)),
    Action.A35 : (action_buy, (Slot.B5, Slot.T0)),
    Action.A36 : (action_buy, (Slot.B5, Slot.T1)),
    Action.A37 : (action_buy, (Slot.B5, Slot.T2)),
    Action.A38 : (action_buy, (Slot.B5, Slot.T3)),
    Action.A39 : (action_buy, (Slot.B5, Slot.T4)),
    Action.A40 : (action_buy, (Slot.B6, Slot.T0)),
    Action.A41 : (action_buy, (Slot.B6, Slot.T1)),
    Action.A42 : (action_buy, (Slot.B6, Slot.T2)),
    Action.A43 : (action_buy, (Slot.B6, Slot.T3)),
    Action.A44 : (action_buy, (Slot.B6, Slot.T4)),

    # Sell a team pet
    Action.A45 : (action_sell, (Slot.T0)),
    Action.A46 : (action_sell, (Slot.T1)),
    Action.A47 : (action_sell, (Slot.T2)),
    Action.A48 : (action_sell, (Slot.T3)),
    Action.A49 : (action_sell, (Slot.T4)),

    # Freeze a shop pet
    Action.A50 : (action_freeze, (Slot.B0)),
    Action.A51 : (action_freeze, (Slot.B1)),
    Action.A52 : (action_freeze, (Slot.B2)),
    Action.A53 : (action_freeze, (Slot.B3)),
    Action.A54 : (action_freeze, (Slot.B4)),
    Action.A55 : (action_freeze, (Slot.B5)),
    Action.A56 : (action_freeze, (Slot.B6)),

    # Roll and end
    Action.A57 : (action_roll, None),
    Action.A58 : (action_end, None)
}

SAP_ACTION_SPACE = list(SAP_ACTION_FUNC.keys())