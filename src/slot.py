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

TEAM_SLOTS = [Slot.T0, Slot.T1, Slot.T2, Slot.T3, Slot.T4]
BUY_SLOTS = [Slot.B0, Slot.B1, Slot.B2, Slot.B3, Slot.B4, Slot.B5, Slot.B6]

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