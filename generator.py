from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import glob
import cv2
import mahotas
import time
import progressbar
IMAGE_PATH = './'
FONT_PATH = './fonts/'
USED_FONT_PATH = './108/'
fonts_name = []
used_fonts_name = []
for root, dirs, files in os.walk(FONT_PATH):
    fonts_name = files

for root, dirs, files in os.walk(USED_FONT_PATH):
    used_fonts_name = files

new_fonts = []
for font in fonts_name:
    if font not in used_fonts_name:
        new_fonts.append(font)
fonts_name = new_fonts
fonts = []
font_path = os.path.join(FONT_PATH, "*.*")
font = list(glob.glob(font_path))
for each in font:
    fnt = ImageFont.truetype(each, int(64*1.75/2.5), 0)
    fonts.append(fnt)
label = open('result.txt')
label = label.read()
label = label.split('\n')
directory = open('remaining.txt')
directory = directory.read()
directory = directory.split('\n')
widgets = ["数据集创建中： ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=len(label), widgets=widgets).start()
bad = open('bad_characters.txt', 'w')
for i in range(len(label)):
    for j in range(len(fonts)):
        canvas = np.zeros(shape=(64, 64), dtype=np.uint8)
        canvas[0:] = 255
        base = Image.fromarray(canvas).convert('RGBA')
        txt = Image.new('RGBA', base.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(txt)
        d.text((3, 3), label[i], font=fonts[j], fill=(0, 0, 0, 255))
        combined = Image.alpha_composite(base, txt)
        (b, g, r, a) = combined.split()
        rgb_img = Image.merge("RGB", (b, g, r))
        img = np.array(rgb_img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        save_path = os.path.join(IMAGE_PATH, str(directory[i]))
        img_name = save_path + '/' + fonts_name[j] + '.png'
        out = img
        row, col = out.shape
        ru = 0
        rb = 0
        cu = 0
        cb = 0
        for r in range(row):
            if out.sum(axis=1)[r]!=255*col:
                ru = r
                break
        for r in range(row-1, 0, -1):
            if out.sum(axis=1)[r]!=255*col:
                rb = r
                break
        for r in range(col):
            if out.sum(axis=0)[r]!=255*row:
                cu = r
                break
        for r in range(col-1, 0, -1):
            if out.sum(axis=0)[r]!=255*row:
                cb = r
                break
        out = out[ru:rb+1, cu:cb+1]
        if out.shape[0]==1 and out.shape[1]==1:
            bad.write(str(label[i]) + '\t' + str(directory[i]) + '\t' + fonts_name[j] + '\n')
            continue
        wanted_x = 64.0
        wanted_y = 64.0
        if out.shape[0] > out.shape[1]:
            y = int(wanted_y)
            x = int(out.shape[1] * wanted_y / out.shape[0])
        else:
            x = int(wanted_x)
            y = int(out.shape[0] * wanted_x / out.shape[1])
        out = cv2.resize(out, (x, y))
        up = int((64 - y) / 2)
        bottom = 64 - y - up
        left = int((64 - x) / 2)
        right = 64 - x - left
        out = cv2.copyMakeBorder(out, up, bottom, left, right, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        out = Image.fromarray(out)
        out.save(img_name)
    pbar.update(i)
pbar.finish()
bad.close()