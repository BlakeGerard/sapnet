from evdev import UInput, InputDevice, ecodes as e
import time

class Mouse:
    def __init__(self, event, position):
        self.dev_ = InputDevice(event)
        self.ui_ = UInput.from_device(self.dev_)
        self.position_ = position

    def position(self):
        return self.position_

    def register_move(self, diff):
        self.ui_.write(e.EV_REL, e.REL_X, diff[0])
        self.ui_.write(e.EV_REL, e.REL_Y, diff[1])
        self.position_[0] += diff[0]
        self.position_[1] += diff[1]

    def register_release(self):
        self.ui_.write(e.EV_KEY, e.BTN_LEFT, 0)

    def register_press(self):
        self.ui_.write(e.EV_KEY, e.BTN_LEFT, 1)

    def register_hold(self):
        self.ui_.write(e.EV_KEY, e.BTN_LEFT, 2)

    def syn(self):
        self.ui_.syn()

    def move(self, dst_diff):
        self.register_move(dst_diff)
        self.syn()

    def click(self):
        self.register_press()
        self.register_release()
        self.syn()

    def move_and_click(dst_diff):
        self.move(dst_diff)
        self.click()

    def drag(self, src_diff, dst_diff):
        self.register_move(src_diff)
        self.register_press()
        self.register_hold()
        self.syn()
        self.register_move(dst_diff)
        self.register_release()
        self.syn()
