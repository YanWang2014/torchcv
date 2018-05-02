'''
imagenet
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
'''
from PIL import Image
import os
import os.path
import torch
import torch.utils.data as data
from torch.utils.data.dataloader import default_collate
import numpy as np
import random
from .transforms import resize
import torchvision
import torchvision.transforms as T

#import cv2


class TextDataset(data.Dataset):

    def __init__(self, label_files=None, image_files=None, mode=None, split_ratio=0.9, transform=None):
        if label_files is not None:
            self.image_names = [name[:-4] for name in os.listdir(label_files)]
        else:
            self.image_names = [name[:-4] for name in os.listdir(image_files)]
        self.transform = transform
        self.label_files = label_files
                
        random.seed(1000)
        random.shuffle(self.image_names)
        
        length = len(self.image_names)
        if mode == 'train':
            self.image_names = self.image_names[: int(split_ratio*length)]
        if mode == 'val':
            self.image_names = self.image_names[int(split_ratio*length):]
        
        self.paths = [image_files + img_name + '.jpg' for img_name in self.image_names]
        self.mode = mode
        
        if mode != 'test':
            self.bboxes = [] 
            self.labels = []
            for img_name in self.image_names:
                bboxes = []
                labels = []
                for line in open(os.path.join(self.label_files, img_name + '.txt'), 'r', encoding='UTF-8').read().splitlines():
                    items = line.split(',')
                    bboxes.append([float(i) for i in items[:8]])
                    labels.append(1)
                self.bboxes.append(torch.Tensor(bboxes))
                self.labels.append(torch.LongTensor(labels))

        print('Finish dataset init')

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        '''Load image.

        Args:
          idx: (int) image index.

        Returns:
          img: (tensor) image tensor.
          boxes: (tensor) bounding box targets.
          labels: (tensor) class label targets.
        '''
        img_name = self.image_names[idx]
        img_path = self.paths[idx]
        
        image = Image.open(img_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        if self.mode == 'test':
            if self.transform:
                image = self.transform(image)
            return image, img_name
        else:
            bboxes = self.bboxes[idx].clone()
            labels = self.labels[idx].clone()
            if self.transform:
                image, bboxes, labels = self.transform(image, bboxes, labels)
            return image, bboxes, labels, img_name

def bbox_collate_fn(batch_list):
    """
    https://zhuanlan.zhihu.com/p/30385675
    输入是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
    
    输出： stacked image tensor, list of string, list of ndarray
    boxes: (tensor) object boxes, sized [#obj,8].
    """
    image_tensor_list = [b[0] for b in batch_list]
    bboxes = [b[1] for b in batch_list]
    labels = [b[2] for b in batch_list]
    name_list = [b[3] for b in batch_list]

    return [default_collate(image_tensor_list), bboxes, labels, name_list]

if __name__ == "__main__":
    #__spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"

    print('start here')

    class AverageMeter(object):
        def __init__(self):
            self.reset()
    
        def reset(self):
            self.val = 0
            self.avg = 0
            self.sum = 0
            self.count = 0
    
        def update(self, val, n=1):
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count
    
    
    batch_size  = 4
    INPUT_WORKERS = 4
    train_label_files = '../../../data/train_1000/txt_1000/'
    train_image_files = '../../../data/train_1000/image_1000/'

    def train_transform(img, boxes,labels):
        #img, boxes = random_paste(img, boxes, max_ratio=4, fill=(123,116,103))
        #img, boxes, labels = random_crop(img, boxes, labels)
        #img, _ = transforms.resize(img, boxes, size=600, random_interpolation=True)
        #img, boxes = random_flip(img, boxes)
        img = T.Resize((512, 512))(img)
        img = T.ToTensor()(img)
        return img, boxes, labels

    transformed_dataset_test = TextDataset(label_files = train_label_files,
                                           image_files = train_image_files,
                                             mode = None,
                                             transform = train_transform)
#                                             transform=transforms.Compose([
#                                                     transforms.Resize((512, 512)),
#                                                     transforms.ToTensor() 
#                                                     ])
#                                               )           
    dataloader = data.DataLoader(transformed_dataset_test, batch_size=batch_size,
                                 shuffle=False, num_workers=INPUT_WORKERS,
                                 collate_fn=bbox_collate_fn)

    
#    #calculate mean and variance
#    mean_meter = AverageMeter()
#    for i, (image, mask, img_name) in enumerate(dataloader):  # nchw
#        if i%10 ==0:
#            print(i)
#        mean_meter.update(image.mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True), image.size(0))  
#    
#    mean = mean_meter.avg
#    print(mean.squeeze())
#    std_meter =  AverageMeter()
#    for i, (image, mask, img_name) in enumerate(dataloader):  # nchw
#        if i%10 ==0:
#            print(i)
#        std_meter.update(((image-mean)**2).mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True), image.size(0))  
#    print(std_meter.avg.squeeze().sqrt())
    
    
    print(len(dataloader))
    img, bboxes, labels, img_name= next(iter(dataloader))
    print(img.size())

    import matplotlib.pyplot as plt
    
    def imshow(inp, title=None):
        inp = inp.numpy().transpose((1, 2, 0))
        plt.figure()
        plt.imshow(inp)
        plt.pause(1)
        
    #imshow(img[0,:,:,:])
    print(img_name[0])
    print(len(img_name))
    print(len(bboxes))
    print(bboxes[0].shape) # bboxes[0] is tensor
    print(bboxes[0].size())
