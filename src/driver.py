#! /bin/python
from train import *
from server import *
from sapnet import SAPNetActorCritic
from os.path import exists

load_path = "models/goddard.h4096.pt"
role = Role.HOST

def main():
    model = SAPNetActorCritic("goddard.h4096")
    if exists(load_path):
        print("Loading goddard")
        model.load(load_path)

    trainer = ActorCriticTrainer(model, role)
    trainer.train()

from mouse import Mouse
import time
if __name__ == "__main__":
    main()

#    mouse = Mouse("/dev/input/event13", (0,0))
#    time.sleep(3)
#    mouse.move_and_click(SLOT_LOC[Slot.T0])
#    time.sleep(1)
#    mouse.move_and_click(SLOT_LOC[Slot.T4])
'''
    time.sleep(2)
    mouse.move_absolute(SLOT_LOC[Slot.B0])
    time.sleep(2)
    mouse.move_absolute(SLOT_LOC[Slot.B1])
    time.sleep(2)
    mouse.move_absolute(SLOT_LOC[Slot.B2])
    time.sleep(2)
    mouse.move_absolute(SLOT_LOC[Slot.B3])
    time.sleep(2)
    mouse.move_absolute(SLOT_LOC[Slot.B4])
    time.sleep(2)
    mouse.move_absolute(SLOT_LOC[Slot.B5])
    time.sleep(2)
    mouse.move_absolute(SLOT_LOC[Slot.B6])
    time.sleep(2)
    mouse.move_absolute(SELL_LOC)
    time.sleep(2)
    mouse.move_absolute(ROLL_LOC)
    time.sleep(2)
    mouse.move_absolute(FREEZE_LOC)
    time.sleep(2)
    mouse.move_absolute(END_LOC)
'''
