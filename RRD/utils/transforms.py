import torch
import random

from PIL import Image


def resize(img, boxes, size, max_size=1000, random_interpolation=False):
    '''Resize the input PIL image to given size.

    If boxes is not None, resize boxes accordingly.

    Args:
      img: (PIL.Image) image to be resized.
      boxes: (tensor) object boxes, sized [#obj,8].
      size: (tuple or int)
        - if is tuple, resize image to the size.
        - if is int, resize the shorter side to the size while maintaining the aspect ratio.
      max_size: (int) when size is int, limit the image longer size to max_size.
                This is essential to limit the usage of GPU memory.
      random_interpolation: (bool) randomly choose a resize interpolation method.

    Returns:
      img: (PIL.Image) resized image.
      boxes: (tensor) resized boxes.

    Example:
    >> img, boxes = resize(img, boxes, 600)  # resize shorter side to 600
    >> img, boxes = resize(img, boxes, (500,600))  # resize image size to (500,600)
    >> img, _ = resize(img, None, (500,600))  # resize image only
    '''
    w, h = img.size
    if isinstance(size, int):
        size_min = min(w,h)
        size_max = max(w,h)
        sw = sh = float(size) / size_min
        if sw * size_max > max_size:
            sw = sh = float(max_size) / size_max
        ow = int(w * sw + 0.5)
        oh = int(h * sh + 0.5)
    else:
        ow, oh = size
        sw = float(ow) / w
        sh = float(oh) / h

    method = random.choice([
        Image.BOX,
        Image.NEAREST,
        Image.HAMMING,
        Image.BICUBIC,
        Image.LANCZOS,
        Image.BILINEAR]) if random_interpolation else Image.BILINEAR
    img = img.resize((ow,oh), method)
    if boxes is not None:
        boxes = boxes * torch.Tensor([sw,sh,sw,sh,sw,sh,sw,sh])
    return img, boxes

def random_flip(img, boxes):
    '''Randomly flip PIL image.

    If boxes is not None, flip boxes accordingly.

    Args:
      img: (PIL.Image) image to be flipped.
      boxes: (tensor) object boxes, sized [#obj,8].
      NOTE: boxes must follow this order:
      x0 x1
      x3 x2

    Returns:
      img: (PIL.Image) randomly flipped image.
      boxes: (tensor) randomly flipped boxes.
    '''
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        w = img.width
        if boxes is not None:
            x0 = w - boxes[:,2]
            y0 = boxes[:,3]
            x1 = w - boxes[:,0]
            y1 = boxes[:,1]
            x2 = w - boxes[:,6]
            y2 = boxes[:,7]
            x3 = w - boxes[:,4]
            y3 = boxes[:,5]
            boxes[:,0] = x0
            boxes[:,1] = y0
            boxes[:,2] = x1
            boxes[:,3] = y1
            boxes[:,4] = x2
            boxes[:,5] = y2
            boxes[:,6] = x3
            boxes[:,7] = y3
            
    return img, boxes

def scale_jitter(img, boxes, sizes, max_size=1400):
    '''Randomly scale image shorter side to one of the sizes.

    If boxes is not None, resize boxes accordingly.

    Args:
      img: (PIL.Image) image to be resized.
      boxes: (tensor) object boxes, sized [#obj,4].
      sizes: (tuple) scale sizes.
      max_size: (int) limit the image longer size to max_size.

    Returns:
      img: (PIL.Image) resized image.
      boxes: (tensor) resized boxes.
    '''
    w, h = img.size
    size_min = min(w,h)
    size_max = max(w,h)
    size = random.choice(sizes)
    sw = sh = float(size) / size_min
    if sw * size_max > max_size:
        sw = sh = float(max_size) / size_max

    ow = int(w * sw + 0.5)
    oh = int(h * sh + 0.5)
    img = img.resize((ow,oh), Image.BILINEAR)

    if boxes is not None:
        boxes = boxes * torch.Tensor([sw,sh,sw,sh])
    return img, boxes

def random_paste(img, boxes, max_ratio=4, fill=0):
    '''Randomly paste the input image on a larger canvas.

    If boxes is not None, adjust boxes accordingly.

    Args:
      img: (PIL.Image) image to be flipped.
      boxes: (tensor) object boxes, sized [#obj,4].
      max_ratio: (int) maximum ratio of expansion.
      fill: (tuple) the RGB value to fill the canvas.

    Returns:
      canvas: (PIL.Image) canvas with image pasted.
      boxes: (tensor) adjusted object boxes.
    '''
    w, h = img.size
    ratio = random.uniform(1, max_ratio)
    ow, oh = int(w*ratio), int(h*ratio)
    canvas = Image.new('RGB', (ow,oh), fill)

    x = random.randint(0, ow - w)
    y = random.randint(0, oh - h)
    canvas.paste(img, (x,y))

    if boxes is not None:
        boxes = boxes + torch.Tensor([x,y,x,y])
    return canvas, boxes


def random_crop(
        img, boxes, labels,
        min_scale=0.3,
        max_aspect_ratio=2.):
    '''Randomly crop a PIL image.

    Args:
      img: (PIL.Image) image.
      boxes: (tensor) bounding boxes, sized [#obj, 4].
      labels: (tensor) bounding box labels, sized [#obj,].
      min_scale: (float) minimal image width/height scale.
      max_aspect_ratio: (float) maximum width/height aspect ratio.

    Returns:
      img: (PIL.Image) cropped image.
      boxes: (tensor) object boxes.
      labels: (tensor) object labels.
    '''
    imw, imh = img.size
    params = [(0, 0, imw, imh)]  # crop roi (x,y,w,h) out
    for min_iou in (0, 0.1, 0.3, 0.5, 0.7, 0.9):
        for _ in range(100):
            scale = random.uniform(min_scale, 1)
            aspect_ratio = random.uniform(
                max(1/max_aspect_ratio, scale*scale),
                min(max_aspect_ratio, 1/(scale*scale)))
            w = int(imw * scale * math.sqrt(aspect_ratio))
            h = int(imh * scale / math.sqrt(aspect_ratio))

            x = random.randrange(imw - w)
            y = random.randrange(imh - h)

            roi = torch.Tensor([[x,y,x+w,y+h]])
            ious = box_iou(boxes, roi)
            if ious.min() >= min_iou:
                params.append((x,y,w,h))
                break

    x,y,w,h = random.choice(params)
    img = img.crop((x,y,x+w,y+h))

    center = (boxes[:,:2] + boxes[:,2:]) / 2
    mask = (center[:,0]>=x) & (center[:,0]<=x+w) \
         & (center[:,1]>=y) & (center[:,1]<=y+h)
    if mask.any():
        boxes = boxes[mask.nonzero().squeeze()] - torch.Tensor([x,y,x,y])
        boxes = box_clamp(boxes, 0,0,w,h)
        labels = labels[mask]
    else:
        boxes = torch.Tensor([[0,0,0,0]])
        labels = torch.LongTensor([0])
    return img, boxes, labels


def random_distort(
    img,
    brightness_delta=32/255.,
    contrast_delta=0.5,
    saturation_delta=0.5,
    hue_delta=0.1):
    '''A color related data augmentation used in SSD.

    Args:
      img: (PIL.Image) image to be color augmented.
      brightness_delta: (float) shift of brightness, range from [1-delta,1+delta].
      contrast_delta: (float) shift of contrast, range from [1-delta,1+delta].
      saturation_delta: (float) shift of saturation, range from [1-delta,1+delta].
      hue_delta: (float) shift of hue, range from [-delta,delta].

    Returns:
      img: (PIL.Image) color augmented image.
    '''
    def brightness(img, delta):
        if random.random() < 0.5:
            img = transforms.ColorJitter(brightness=delta)(img)
        return img

    def contrast(img, delta):
        if random.random() < 0.5:
            img = transforms.ColorJitter(contrast=delta)(img)
        return img

    def saturation(img, delta):
        if random.random() < 0.5:
            img = transforms.ColorJitter(saturation=delta)(img)
        return img

    def hue(img, delta):
        if random.random() < 0.5:
            img = transforms.ColorJitter(hue=delta)(img)
        return img

    img = brightness(img, brightness_delta)
    if random.random() < 0.5:
        img = contrast(img, contrast_delta)
        img = saturation(img, saturation_delta)
        img = hue(img, hue_delta)
    else:
        img = saturation(img, saturation_delta)
        img = hue(img, hue_delta)
        img = contrast(img, contrast_delta)
    return img
