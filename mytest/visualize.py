# -- coding: utf-8 --
"""
@Time：2022-10-15 10:30
@Author：zstar
@File：visualize.py
@Describe：将生成的labels添加到图片中可视化，输出visualize_output
"""

import os
import shutil
from pathlib import Path
import numpy as np
import cv2

# 修改输入图片文件夹
img_folder = "origin_img/"
img_list = os.listdir(img_folder)
img_list.sort()
# 修改输入标签文件夹
label_folder = "labels/"
label_list = os.listdir(label_folder)
label_list.sort()
# 输出图片文件夹位置
path = os.getcwd()
output_folder = path + '/' + str("visualize_output")

classes = ['leibi1', 'leibi2', 'leibi3']
colormap = [(0, 255, 0), (132, 112, 255), (0, 191, 255)]  # 色盘，可根据类别添加新颜色


# 坐标转换
def xywh2xyxy(x, w1, h1, img):
    label, x, y, w, h = x
    label = int(label)
    # print("原图宽高:\nw1={}\nh1={}".format(w1, h1))
    # 边界框反归一化
    x_t = x * w1
    y_t = y * h1
    w_t = w * w1
    h_t = h * h1
    # print("反归一化后输出：\n第一个:{}\t第二个:{}\t第三个:{}\t第四个:{}\t\n\n".format(x_t, y_t, w_t, h_t))
    # 计算坐标
    top_left_x = x_t - w_t / 2
    top_left_y = y_t - h_t / 2
    bottom_right_x = x_t + w_t / 2
    bottom_right_y = y_t + h_t / 2
    # print('标签:{}'.format(labels[int(label)]))
    # print("左上x坐标:{}".format(top_left_x))
    # print("左上y坐标:{}".format(top_left_y))
    # print("右下x坐标:{}".format(bottom_right_x))
    # print("右下y坐标:{}".format(bottom_right_y))
    # 绘制矩形框
    cv2.rectangle(img, p1, p2, colormap[1], thickness=2, lineType=cv2.LINE_AA)
    label = labels[label]
    if label:
        w, h = cv2.getTextSize(label, 0, fontScale=2 / 3, thickness=2)[0]  # text width, height
        outside = p1[1] - h - 3 >= 0  # label fits outside box
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        # 绘制矩形框填充
        cv2.rectangle(img, p1, p2, colormap[1], -1, cv2.LINE_AA)
        # 绘制标签
        cv2.putText(img, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0, 2 / 3, colormap[2],
                    thickness=2, lineType=cv2.LINE_AA)
    """
    # (可选)给不同目标绘制不同的颜色框
    if int(label) == 0:
       cv2.rectangle(img, (int(top_left_x), int(top_left_y)), (int(bottom_right_x), int(bottom_right_y)), (0, 255, 0), 2)
    elif int(label) == 1:
       cv2.rectangle(img, (int(top_left_x), int(top_left_y)), (int(bottom_right_x), int(bottom_right_y)), (255, 0, 0), 2)
    """
    return img


if __name__ == '__main__':
    # 创建输出文件夹
    if Path(output_folder).exists():
        shutil.rmtree(output_folder)
    os.mkdir(output_folder)
    for i in range(len(img_list)):
        image_path = img_folder + "/" + img_list[i]
        label_path = label_folder + "/" + label_list[i]
        # 读取图像文件
        img = cv2.imread(str(image_path))
        h, w = img.shape[:2]
        # 读取 labels
        with open(label_path, 'r') as f:
            lb = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)
        # 绘制每一个目标
        for x in lb:
            # 反归一化并得到左上和右下坐标，画出矩形框
            img = xywh2xyxy(x, w, h, img)
        """
        # 直接查看生成结果图
        cv2.imshow('show', img)
        cv2.waitKey(0)
        """
        cv2.imwrite(output_folder + '/' + '{}.png'.format(image_path.split('/')[-1][:-4]), img)
