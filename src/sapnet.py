from enum import Enum
from numpy import floor, ceil

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

SLOT_HEIGHT = 0.128
SLOT_WIDTH  = 0.131

ROW_0 = 0.148
ROW_1 = ROW_0 + SLOT_HEIGHT
ROW_2 = 0.300
ROW_3 = ROW_2 + SLOT_HEIGHT

COL_0 = 0.423
COL_1 = COL_0 + 1.0 * SLOT_WIDTH
COL_2 = COL_0 + 2.0 * SLOT_WIDTH
COL_3 = COL_0 + 3.0 * SLOT_WIDTH
COL_4 = COL_0 + 4.0 * SLOT_WIDTH
COL_5 = COL_0 + 5.0 * SLOT_WIDTH
COL_6 = COL_0 + 6.0 * SLOT_WIDTH
COL_7 = COL_0 + 7.0 * SLOT_WIDTH

SLOT_LOC = {
    Slot.T0 : (ROW_0, ROW_1, COL_0, COL_1),
    Slot.T1 : (ROW_0, ROW_1, COL_1, COL_2),
    Slot.T2 : (ROW_0, ROW_1, COL_2, COL_3),
    Slot.T3 : (ROW_0, ROW_1, COL_3, COL_4),
    Slot.T4 : (ROW_0, ROW_1, COL_4, COL_5),
    Slot.B0 : (ROW_2, ROW_3, COL_0, COL_1),
    Slot.B1 : (ROW_2, ROW_3, COL_1, COL_2),
    Slot.B2 : (ROW_2, ROW_3, COL_2, COL_3),
    Slot.B3 : (ROW_2, ROW_3, COL_3, COL_4),
    Slot.B4 : (ROW_2, ROW_3, COL_4, COL_5),
    Slot.B5 : (ROW_2, ROW_3, COL_5, COL_6),
    Slot.B6 : (ROW_2, ROW_3, COL_6, COL_7),
}

def get_slot_img(shop_img, slot):
    h = float(shop_img.shape[0])
    w = float(shop_img.shape[1])
    loc = SLOT_LOC[slot]
    return shop_img[int(floor(loc[0] * w)) : int(ceil(loc[1] * w)),
				    int(floor(loc[2] * h)) : int(ceil(loc[3] * h))]