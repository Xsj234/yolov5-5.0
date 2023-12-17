# -- coding: utf-8 --
"""
@Time：2022-10-15 9:36
@Author：zstar
@File：crop_merge.py
@Describe：将crops文件夹图片根据类别融合成labels
"""

import os
from pathlib import Path
import shutil

classes = ['person', 'bus', 'fire hydrant', 'tie']
crops_path = "../runs/detect/exp19/crops"  # 'crops'
labels_path = Path('../runs/detect/exp19/labels/bus.txt')

if __name__ == '__main__':
    # 由于后续是追加写入txt，因此先要删除labels，再进行创建
    if labels_path.exists():
        shutil.rmtree(labels_path)
    os.mkdir(labels_path)
    for dir_class in os.listdir(crops_path):
        index = classes.index(dir_class)  # 根据类别顺序分配0-5
        for img in os.listdir(Path(f"{crops_path}/{dir_class}")):
            img = img[:-4]  # 去除文件名后缀
            parts = img.split('__')
            img_name = parts[0]
            x = float(parts[1])
            y = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
            line = (index, x, y, w, h)
            with open('labels' + '/' + img_name + '.txt', mode='a') as f:
                f.write(('%g ' * len(line)).rstrip() % line + '\n')
