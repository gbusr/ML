#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
import sys
import time
import cv2
import dlib

from keras.preprocessing import image as imagekeras
from keras.models import load_model
from PIL import Image, ImageDraw, ImageFont

size = 150
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_face_trained_model.h5'

# 类别编码转换为中文名称返回
def return_name(codelist):
    names = ['樊胜美', '关雎尔', '邱莹莹']
    for it in range(0, len(codelist), 1):
        if int(codelist[it]) == 1.0:
            return names[it]

# 类别编码转换为英文名称返回
def return_name_en(codelist):
    names = ['fsm', 'gje', 'qyy']
    for it in range(0, len(codelist), 1):
        if int(codelist[it]) == 1.0:
            return names[it]

# 区分和标记视频中截图的人脸
def face_rec():
    global  image_ouput
    model = load_model(os.path.join(save_dir, model_name))
    camera = cv2.VideoCapture("2.mp4") # 视频
    # camera = cv2.VideoCapture(0) # 摄像头

    while (True):
        read, img = camera.read()
        try:
            # 未截取视频图片结束本次循环
            if not (type(img) is np.ndarray):
                continue
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 图片转为灰度图
        except:
            print("Unexpected error:", sys.exc_info()[0])
            break

        # 使用detector进行人脸检测
        # 使用dlib自带的frontal_face_detector作为我们的特征提取器
        detector = dlib.get_frontal_face_detector()
        dets = detector(gray_img, 1) # 提取截图中所有人脸

        facelist = []
        for i, d in enumerate(dets): # 依次区分截图中的人脸
            x1 = d.top() if d.top() > 0 else 0
            y1 = d.bottom() if d.bottom() > 0 else 0
            x2 = d.left() if d.left() > 0 else 0
            y2 = d.right() if d.right() > 0 else 0

            img = cv2.rectangle(img, (x2, x1), (y2, y1), (255, 0, 0), 2) # 人脸画框

            face = img[x1:y1, x2:y2]
            face = cv2.resize(face, (size, size))

            x_input = np.expand_dims(face, axis=0)
            prey = model.predict(x_input)
            print(prey, 'prey')

            facelist.append([d,  return_name(prey[0])]) # 存储一张图中多张人脸坐标和标记（姓名）

        cv2_im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # cv2和PIL中颜色的hex码的储存顺序不同
        pil_im = Image.fromarray(cv2_im)

        draw = ImageDraw.Draw(pil_im)  # 括号中为需要打印的cqanvas，这里就是在图片上直接打印
        font = ImageFont.truetype("simhei.ttf", 20, encoding="utf-8")  # 第一个参数为字体文件路径，第二个为字体大小

        try:
            for i in facelist:
                # 人脸标记写入图片，第一个参数为打印的坐标，第二个为打印的文本，第三个为字体颜色，第四个为字体
                draw.text((i[0].left() + int((i[0].right() - i[0].left()) / 2 - len(i[1]) * 10), i[0].top() - 20), i[1],
                          (255, 0, 0), font=font)
        except:
            print("Unexpected error:", sys.exc_info()[0])
            continue

        # PIL图片转换为cv2图片
        cv2_char_img = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)
        # 显示图片
        cv2.imshow("camera", cv2_char_img)
        if cv2.waitKey(1) & 0xff == ord("q"):
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    face_rec()
