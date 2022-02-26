from enum import Enum 

SAP_WINDOW_L = 0
SAP_WINDOW_T = 30
SAP_WINDOW_W = 960
SAP_WINDOW_H = 600

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

#T0_REGION = (202, 256, 79, 45)
#T1_REGION = (284, 256, 79, 45)
#T2_REGION = (361, 256, 79, 45)
#T3_REGION = (441, 256, 79, 45)
#T4_REGION = (522, 256, 79, 45)
