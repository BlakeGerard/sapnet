from state import StateServer
import numpy as np
import cv2 as cv

srv = StateServer()
srv.start()
state = srv.get_state()

cv.imshow("State", state)
cv.waitKey(0)
cv.destroyAllWindows()
cv.waitKey(1)