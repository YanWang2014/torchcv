from __future__ import print_function

import sys
sys.path.append('../')
import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.utils.data as data

import torchvision
import torchvision.transforms as T

from PIL import Image
from torch.autograd import Variable

#from torchcv.models.fpnssd import FPNSSD512
from torchcv.models.ssd import SSD300, SSD512, SSDBoxCoder

from torchcv.loss import SSDLoss
from utils import text_dataset
from utils import transforms

lr =1e-4
img_size = 512
batch_size  = 32

train_label_files = '../../data/train_1000/txt_1000/'
train_image_files = '../../data/train_1000/image_1000/'

checkpoints = 'pths/ckpt.pth'
resume = False
INPUT_WORKERS = 16


# Model
print('==> Building model..')
net = SSD512(num_classes=2)
#net = FPNSSD512(num_classes=2)
#net.load_state_dict(torch.load(args.model))
best_loss = float('inf')  # best test loss
start_epoch = 0  # start from epoch 0 or last epoch
if resume:
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load(checkpoints)
    net.load_state_dict(checkpoint['net'])
    best_loss = checkpoint['loss']
    start_epoch = checkpoint['epoch']

# Dataset
print('==> Preparing dataset..')
box_coder = SSDBoxCoder(net)
def train_transform(img, boxes, labels):
    img, boxes = transforms.resize(img, boxes, size=(img_size, img_size), random_interpolation=False)
    img = T.Compose([
        T.ToTensor(),
        T.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
    ])(img)
    boxes, labels = box_coder.encode(boxes, labels)
    return img, boxes, labels

def val_transform(img, boxes, labels):
    img, boxes = transforms.resize(img, boxes, size=(img_size, img_size), random_interpolation=False)
    img = T.Compose([
        T.ToTensor(),
        T.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
    ])(img)
    boxes, labels = box_coder.encode(boxes, labels)
    return img, boxes, labels

trainset = text_dataset.TextDataset(label_files = train_label_files,
                                    image_files = train_image_files,
                                    mode = 'train',
                                    transform=train_transform)

valset = text_dataset.TextDataset(label_files = train_label_files,
                                   image_files = train_image_files,
                                   mode = 'val',
                                   transform=val_transform)

trainloader =  data.DataLoader(trainset, batch_size=batch_size,
                               shuffle=True, num_workers=INPUT_WORKERS
                               #collate_fn=text_dataset.bbox_collate_fn
                               )
valloader =  data.DataLoader(valset, batch_size=batch_size,
                               shuffle=False, num_workers=INPUT_WORKERS
                               #collate_fn=text_dataset.bbox_collate_fn
                               )

print(len(trainloader))
print(len(valloader))
img, bboxes, labels, img_name= next(iter(trainloader))
print(img.size())
print(bboxes.size())
print(labels.size())

net = torch.nn.DataParallel(net)#, device_ids=[2,3,4,5])
cudnn.benchmark = True
net.cuda()
criterion = SSDLoss(num_classes=2)
criterion.cuda()
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    for batch_idx, (inputs, loc_targets, cls_targets, names) in enumerate(trainloader):
        #print(batch_idx/len(trainloader))
        inputs = Variable(inputs.cuda())
        loc_targets = Variable(loc_targets.cuda())
        cls_targets = Variable(cls_targets.cuda())

        optimizer.zero_grad()
        loc_preds, cls_preds = net(inputs)

        loss = criterion(loc_preds, loc_targets, cls_preds, cls_targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        print('train_loss: %.3f | avg_loss: %.3f [%d/%d]'
              % (loss.data[0], train_loss/(batch_idx+1), batch_idx+1, len(trainloader)))

# Test
def test(epoch):
    print('\nTest')
    net.eval()
    test_loss = 0
    for batch_idx, (inputs, loc_targets, cls_targets, names) in enumerate(valloader):
        inputs = Variable(inputs.cuda(), volatile=True)
        loc_targets = Variable(loc_targets.cuda())
        cls_targets = Variable(cls_targets.cuda())

        loc_preds, cls_preds = net(inputs)
        loss = criterion(loc_preds, loc_targets, cls_preds, cls_targets)
        test_loss += loss.data[0]
        print('test_loss: %.3f | avg_loss: %.3f [%d/%d]'
              % (loss.data[0], test_loss/(batch_idx+1), batch_idx+1, len(valloader)))

    # Save checkpoint
    global best_loss
    test_loss /= len(valloader)
    if test_loss < best_loss:
        print('Saving..')
        state = {
            'net': net.module.state_dict(),
            'loss': test_loss,
            'epoch': epoch,
        }
        if not os.path.isdir(os.path.dirname(checkpoints)):
            os.mkdir(os.path.dirname(checkpoints))
        torch.save(state, checkpoints)
        best_loss = test_loss


for epoch in range(start_epoch, start_epoch+200):
    train(epoch)
    test(epoch)

