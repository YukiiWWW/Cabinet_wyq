# coding:utf-8
import easyocr
import cv2
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont

t1 = time.time()
reader = easyocr.Reader(['ch_sim','en'])
result1 = reader.readtext('test1.jpg')
t2 = time.time()
result2 = reader.readtext('test1.jpg')
t3 = time.time()

print("the time is :%f s"%(t2-t1))
print("the time is :%f s"%(t3-t2))
print(result1)
print(result2)
# img_path = "./deployment/test1.jpg"

# img = cv2.imread(img_path)
# boxes = []
# for item in result:
#     print(str(item[0]))
#     boxes.append(item[0])

# img_PIL = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# draw = ImageDraw.Draw(img_PIL)
# rgb = (0, 255, 0)
# for i in range(len(boxes)):
#     box = boxes[i]
#     x1 = box[0][0]
#     y1 = box[0][1]
#     x2 = box[2][0]
#     y2 = box[2][1]
#     draw.rectangle((x1, y1, x2, y2), outline=rgb)

#img2 = utils.draw_bboxes(img,bbox)
# img = cv2.cvtColor(np.asarray(img_PIL), cv2.COLOR_RGB2BGR)
# cv2.imshow('img',img)
# cv2.waitKey(0)
# print(result)