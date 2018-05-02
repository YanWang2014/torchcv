# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 16:22:44 2018

@author: Beyond

把gt贴到图片上看看

X1，Y1，X2，Y2，X3，Y3，X4，Y4
"""

import os
import pandas as pd

train_label_files = 'train_part1_20180316/train_1000/txt_1000/'
train_image_files = 'train_part1_20180316/train_1000/image_1000/'

id_to_file = [name[:-4] for name in os.listdir(train_label_files)]
#file_to_id = dict((v, k) for k, v in enumerate(id_to_file))

all_labels = []
for img_id, file in enumerate(id_to_file):
    for line in open(os.path.join(train_label_files, file + '.txt'), 'r', encoding='UTF-8').read().splitlines():
        items = line.split(',')
#        if items[-1] == '###':
#            continue
        entry = [img_id] + [float(i) for i in items[:8]]
        entry.append(items[-1])
        all_labels.append(entry)

df_all_labels = pd.DataFrame(all_labels, columns=['img','x1','y1','x2','y2','x3','y3','x4','y4','text'])    


    
import cv2
import random
from matplotlib import pyplot as plt

img_id = random.randint(0, len(id_to_file)-1)
#for img_id, name in enumerate(id_to_file):
img = cv2.imread(os.path.join(train_image_files, id_to_file[img_id]+'.jpg'))
for _, label in df_all_labels[df_all_labels['img']==img_id].iterrows():
    cv2.line(img, (round(label['x1']), round(label['y1'])), (round(label['x2']), round(label['y2'])), (255, 0, 0), 2)
    cv2.line(img, (round(label['x2']), round(label['y2'])), (round(label['x3']), round(label['y3'])), (255, 255, 0), 2)
    cv2.line(img, (round(label['x3']), round(label['y3'])), (round(label['x4']), round(label['y4'])), (0, 0, 255), 2)
    cv2.line(img, (round(label['x4']), round(label['y4'])), (round(label['x1']), round(label['y1'])), (100, 100, 100), 2)
plt.subplots(figsize=(10,10))
plt.imshow(img)
plt.title(str(img_id))
#    cv2.imwrite(train_image_files+name+'2.jpg',img)