from evdev import UInput, InputDevice, ecodes as e
import time

KEY_UP = 0
KEY_DOWN = 1
KEY_HOLD = 2

class Mouse:
    def __init__(self, event, position):
        self.dev_ = InputDevice(event)
        self.ui_ = UInput.from_device(self.dev_)
        self.position_ = [0,0]

    def position(self):
        return self.position_

    def pos_diff(self, dst):
        print("Position: ", self.position_)
        dx = round(dst[0] - self.position_[0])
        dy = round(dst[1] - self.position_[1])
        print("Diff: ", (dx, dy))
        return (dx, dy)

    def register_move(self, diff):
        self.ui_.write(e.EV_REL, e.REL_X, diff[0])
        self.ui_.write(e.EV_REL, e.REL_Y, diff[1])
        self.position_[0] += diff[0]
        self.position_[1] += diff[1]
        print("New Position: ", self.position_)

    def register_release(self):
        self.ui_.write(e.EV_KEY, e.BTN_LEFT, KEY_UP)

    def register_press(self):
        self.ui_.write(e.EV_KEY, e.BTN_LEFT, KEY_DOWN)

    def register_hold(self):
        self.ui_.write(e.EV_KEY, e.BTN_LEFT, KEY_HOLD)

    def syn(self):
        self.ui_.syn()

    def move_absolute(self, dst):
        dst_diff = self.pos_diff(dst)
        self.register_move(dst_diff)
        self.syn()

    def click(self):
        self.register_press()
        self.syn()
        self.register_release()
        self.syn()

    def move_and_click(self, dst):
        print("Moving to: ", dst)
        self.move_absolute(dst)
        self.click()

    def drag(self, src, dst):
        self.move_absolute(src)
        self.syn()
        self.register_press()
        self.register_hold()
        self.syn()
        time.sleep(0.5)
        self.move_absolute(dst)
        self.syn()
        self.register_release()
        self.syn()
