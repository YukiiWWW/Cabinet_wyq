import sys
import os
import time
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2


def draw_bboxes(img, boxes, class_names=None, color=None):

    colors = np.array([[1,0,1],[0,0,1],[0,1,1],[0,1,0],[1,1,0],[1,0,0]]);
    def get_color(c, x, max_val):
        ratio = float(x)/max_val * 5
        i = int(math.floor(ratio))
        j = int(math.ceil(ratio))
        ratio = ratio - i
        r = (1-ratio) * colors[i][c] + ratio*colors[j][c]
        return int(r*255)

    width = img.shape[1]
    height = img.shape[0]
    # cv2 to PIL
    # font = ImageFont.truetype('SIMKAI.TTF', 20)
    img_PIL = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_PIL)
    for i in range(len(boxes)):
        box = boxes[i]
        x1 = int(((box[0]) * width))
        y1 = int(((box[1]) * height))
        x2 = int(((box[0] + box[2]) * width))
        y2 = int(((box[1] + box[3]) * height))

        if color:
            rgb = color
        else:
            rgb = (255, 0, 0)
        if len(box) >= 5 and class_names:
            cls_id = int(box[-1])
            # print('%s: %f' % (class_names[cls_id], cls_conf))
            classes = len(class_names)
            offset = cls_id * 123457 % classes
            red   = get_color(2, offset, classes)
            green = get_color(1, offset, classes)
            blue  = get_color(0, offset, classes)
            if color is None:
                rgb = (red, green, blue)
            # draw.text((x1, y2), class_names[cls_id], font=font, fill=rgb)
        draw.rectangle((x1, y1, x2, y2), outline=rgb)
    img = cv2.cvtColor(np.asarray(img_PIL), cv2.COLOR_RGB2BGR)
    return img


def draw_text(img, texts):
    width = img.shape[1]
    height = img.shape[0]
    # cv2 to PIL
    font = ImageFont.truetype('SIMKAI.TTF', 20)
    img_PIL = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_PIL)
    for text in texts:
        draw.text((text[0]*width, text[1]*height), ('{}'.format(text[2])), font=font, fill=text[3])
    img = cv2.cvtColor(np.asarray(img_PIL), cv2.COLOR_RGB2BGR)
    return img
