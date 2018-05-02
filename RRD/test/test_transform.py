import sys
sys.path.append('../')
sys.path.append('../../')

from PIL import Image
from utils import transforms
from utils import text_dataset
import torchvision.transforms as T
import torch.utils.data as data


def train_transform(img, boxes, labels):
    #img, boxes = random_paste(img, boxes, max_ratio=4, fill=(123,116,103))
    #img, boxes, labels = random_crop(img, boxes, labels)
    img, boxes = transforms.resize(img, boxes, size=(512, 512), random_interpolation=False)
    #img, boxes = transforms.random_flip(img, boxes)
    #img = T.Resize((512,512))(img)
    img = T.ToTensor()(img)
    return img, boxes, labels


batch_size  = 4
INPUT_WORKERS = 4
train_label_files = '../train_part1_20180316/train_1000/txt_1000/'
train_image_files = '../train_part1_20180316/train_1000/image_1000/'

transformed_dataset_test = text_dataset.TextDataset(label_files = train_label_files,
                                       image_files = train_image_files,
                                         mode = 'train',
                                         transform=train_transform)

dataloader = data.DataLoader(transformed_dataset_test, batch_size=batch_size,
                             shuffle=True, num_workers=INPUT_WORKERS,
                             collate_fn=text_dataset.bbox_collate_fn)

print(len(dataloader))
img, bboxes, labels, img_name= next(iter(dataloader))
img = img[0,:,:,:].numpy().transpose((1, 2, 0)) # HWC
bbox = bboxes[0].numpy() # (N,8)

import cv2
from matplotlib import pyplot as plt

img = cv2.resize(img, img.shape[:2])
for i in range(bbox.shape[0]):
    cv2.line(img, (round(bbox[i,0]), round(bbox[i,1])), (round(bbox[i,2]), round(bbox[i,3])), (255, 0, 0), 2)
    cv2.line(img, (round(bbox[i,2]), round(bbox[i,3])), (round(bbox[i,4]), round(bbox[i,5])), (255, 255, 0), 2)
    cv2.line(img, (round(bbox[i,4]), round(bbox[i,5])), (round(bbox[i,6]), round(bbox[i,7])), (0, 0, 255), 2)
    cv2.line(img, (round(bbox[i,6]), round(bbox[i,7])), (round(bbox[i,0]), round(bbox[i,1])), (100, 100, 100), 2)
plt.subplots(figsize=(10,10))
plt.imshow(img)
