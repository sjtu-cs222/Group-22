import os
import cv2
import numpy as np


wanted_x = 64.0
wanted_y = 64.0

savePath = "images/"
base = "raw/"
for im in os.listdir(base):
    print(im)
    imPath = base + im
    pic = cv2.imread(imPath, cv2.IMREAD_GRAYSCALE)
    
    bounds = [0]*4
    for i in range(pic.shape[0]):
        if np.min(pic[i, :]) == 0:
            bounds[0] = i
            break
    for i in range(pic.shape[0] - 1, 0, -1):
        if np.min(pic[i, :]) == 0:
            bounds[1] = i + 1
            break
    for i in range(pic.shape[1]):
        if np.min(pic[:, i]) == 0:
            bounds[2] = i
            break
    for i in range(pic.shape[1] - 1, 0, -1):
        if np.min(pic[:, i]) == 0:
            bounds[3] = i + 1
            break
    pic = pic[bounds[0]:bounds[1], bounds[2]:bounds[3]]

    try:
        if pic.shape[0] > pic.shape[1]:
            y = int(wanted_y)
            x = int(pic.shape[1] * wanted_y / pic.shape[0])
        else:
            x = int(wanted_x)
            y = int(pic.shape[0] * wanted_x / pic.shape[1])
        pic = cv2.resize(pic, (x, y), interpolation=cv2.INTER_AREA)
        up = int((64 - y) / 2)
        bottom = 64 - y - up
        left = int((64 - x) / 2)
        right = 64 - x - left

        pic = cv2.copyMakeBorder(pic, up, bottom, left, right, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        cv2.imwrite(savePath + im, pic)
    except:
        print(im, " conversion failed.")
        continue
print("Done.")
    