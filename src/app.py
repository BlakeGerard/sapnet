import json
import cv2 as cv
import numpy as np
import sapnet as sp

shop_img = cv.imread("resources/shop.png", cv.IMREAD_COLOR)
slot_t4 = sp.get_slot_img(shop_img, sp.Slot.B0)
pet_img = cv.imread("resources/fish.png", cv.IMREAD_UNCHANGED)
pet_img = cv.flip(pet_img, 1)
pet_mask = pet_img[:,:,3:]

pet_img = cv.imread("resources/fish.png", cv.IMREAD_COLOR)
result = cv.matchTemplate(image=slot_t4, templ=pet_img, method=cv.TM_CCORR_NORMED, mask=pet_mask)

min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
print("Best match: %s" % str(max_loc))
print("Confidence: %s" % str(max_val))

threshold = 0.0
if max_val >= threshold:
 	sheep_h = pet_img.shape[0]
 	sheep_w = pet_img.shape[1]
 	top_left = max_loc
 	bottom_right = (top_left[0] + sheep_w, top_left[1] + sheep_h)
 	cv.rectangle(slot_t4, top_left, bottom_right, color=(0,0,255),
 		thickness = 2, lineType=cv.LINE_4)
 	cv.imshow('Result', slot_t4)
 	cv.waitKey()
