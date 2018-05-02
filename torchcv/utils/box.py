import torch
import shapely
from shapely.geometry import Polygon,MultiPoint
import numpy as np
import multiprocessing
import itertools

def change_box_order(boxes, order):
    '''Change box order between (xmin,ymin,xmax,ymax) and (xcenter,ycenter,width,height).

    Args:
      boxes: (tensor) bounding boxes, sized [N,4] or [N,8].
      order: (str) either 'xyxy2xywh' or 'xywh2xyxy' or 'xywh2xyxyxyxy'.

    Returns:
      (tensor) converted bounding boxes, sized [N,4] or [N,8].
    '''
    assert order in ['xyxy2xywh','xywh2xyxy', 'xywh2xyxyxyxy']
    
    a = boxes[:,:2]
    b = boxes[:,2:]
    if order == 'xyxy2xywh':
        return torch.cat([(a+b)/2,b-a], 1)
    if order == 'xywh2xyxy':
        return torch.cat([a-b/2,a+b/2], 1)
    if order == 'xywh2xyxyxyxy':
        #a: N, (cx,cy) b: N, (w,h)
        c = torch.cat([boxes[:,2:3], -boxes[:,3:]], 1) # N, (w,-h)
        # 1 2
        # 4 3
        # y is vertical
        return torch.cat([a-b/2, a+c/2, a+b/2, a-c/2], 1)

def box_clamp(boxes, xmin, ymin, xmax, ymax):
    '''Clamp boxes.

    Args:
      boxes: (tensor) bounding boxes of (xmin,ymin,xmax,ymax), sized [N,4].
      xmin: (number) min value of x.
      ymin: (number) min value of y.
      xmax: (number) max value of x.
      ymax: (number) max value of y.

    Returns:
      (tensor) clamped boxes.
    '''
    boxes[:,0].clamp_(min=xmin, max=xmax)
    boxes[:,1].clamp_(min=ymin, max=ymax)
    boxes[:,2].clamp_(min=xmin, max=xmax)
    boxes[:,3].clamp_(min=ymin, max=ymax)
    return boxes

def box_select(boxes, xmin, ymin, xmax, ymax):
    '''Select boxes in range (xmin,ymin,xmax,ymax).

    Args:
      boxes: (tensor) bounding boxes of (xmin,ymin,xmax,ymax), sized [N,4].
      xmin: (number) min value of x.
      ymin: (number) min value of y.
      xmax: (number) max value of x.
      ymax: (number) max value of y.

    Returns:
      (tensor) selected boxes, sized [M,4].
      (tensor) selected mask, sized [N,].
    '''
    mask = (boxes[:,0]>=xmin) & (boxes[:,1]>=ymin) \
         & (boxes[:,2]<=xmax) & (boxes[:,3]<=ymax)
    boxes = boxes[mask.nonzero().squeeze(),:]
    return boxes, mask

def box_iou(box1, box2):
    '''Compute the intersection over union of two set of boxes.

    The box order must be (xmin, ymin, xmax, ymax).

    Args:
      box1: (tensor) bounding boxes, sized [N,4].
      box2: (tensor) bounding boxes, sized [M,4].

    Return:
      (tensor) iou, sized [N,M].

    Reference:
      https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py
    '''
    N = box1.size(0)
    M = box2.size(0)

    lt = torch.max(box1[:,None,:2], box2[:,:2])  # [N,M,2]
    rb = torch.min(box1[:,None,2:], box2[:,2:])  # [N,M,2]

    wh = (rb-lt).clamp(min=0)      # [N,M,2]
    inter = wh[:,:,0] * wh[:,:,1]  # [N,M]

    area1 = (box1[:,2]-box1[:,0]) * (box1[:,3]-box1[:,1])  # [N,]
    area2 = (box2[:,2]-box2[:,0]) * (box2[:,3]-box2[:,1])  # [M,]
    iou = inter / (area1[:,None] + area2 - inter)
    return iou

def quadrilateral_area(box):
    '''Compute the area of quadrilateral
    
    The box order must be ('x1','y1','x2','y2','x3','y3','x4','y4').
    The point order is:
    1 2
    4 3    
    
    Args:
      box: (tensor) bounding boxes, sized [N,8].

    Return:
      (tensor) area, sized [N,].
      
    Reference:
      https://en.wikipedia.org/wiki/Shoelace_formula
      https://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates
      https://github.com/MhLiao/TextBoxes_plusplus/blob/master/examples/text/nms.py
    '''
    
    return

def polygon_area_from_tensor(box):
    polygon_points = box.numpy().reshape(4, 2)
    polygon = Polygon(polygon_points).convex_hull
    return polygon.area

def quadrilateral_iou(box1, box2):
    '''Compute the intersection over union of two set of boxes.

    The box order must be ('x1','y1','x2','y2','x3','y3','x4','y4').
    1 2
    4 3

    Args:
      box1: (tensor) bounding boxes, sized [N,8].
      box2: (tensor) bounding boxes, sized [M,8].

    Return:
      (tensor) iou, sized [N,M].

    Reference:
      https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py
    '''
    N = box1.size(0)
    M = box2.size(0)
    box1_np = box1.numpy()
    box2_np = box2.numpy()
    
    print('iou')
    print(N)
    print(M)
    
    #pool = multiprocessing.Pool()#(processes=16)
    tasks = [polygon_iou(box1_np[i,:], box2_np[j,:]) for (i,j) in itertools.product(range(N), range(M))]
    #iou = pool.starmap(polygon_iou, tasks)
    iou = np.stack(tasks)
    
    return torch.Tensor(iou).view(N,M)

def polygon_iou(list1, list2):
    """
    Intersection over union between two shapely polygons.
    
    for both Tensor and numpy array
    """
    if isinstance(list1, torch.Tensor):
        polygon_points1 = list1.numpy().reshape(4, 2)
        polygon_points2 = list2.numpy().reshape(4, 2)
    else:
        polygon_points1 = list1.reshape(4, 2)
        polygon_points2 = list2.reshape(4, 2)    
    poly1 = Polygon(polygon_points1).convex_hull
    poly2 = Polygon(polygon_points2).convex_hull
    union_poly = np.concatenate((polygon_points1,polygon_points2))
    if not poly1.intersects(poly2): # this test is fast and can accelerate calculation
        iou = 0
    else:
        try:
            inter_area = poly1.intersection(poly2).area
            #union_area = poly1.area + poly2.area - inter_area
            union_area = MultiPoint(union_poly).convex_hull.area
            if union_area == 0:
                return 0
            iou = float(inter_area) / union_area
        except shapely.geos.TopologicalError:
            print('shapely.geos.TopologicalError occured, iou set to 0')
            iou = 0
    return iou

def box_nms(bboxes, scores, threshold=0.5, mode='union'):
    '''Non maximum suppression.

    Args:
      bboxes: (tensor) bounding boxes, sized [N,8].
      scores: (tensor) confidence scores, sized [N,].
      threshold: (float) overlap threshold.
      mode: (str) 'union' or 'min'.

    Returns:
      keep: (tensor) selected indices.

    Reference:
      https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/nms/py_cpu_nms.py
    '''
    _, order = scores.sort(0, descending=True)
    bboxes_np = bboxes.numpy()
    
    keep = []
    while order.numel() > 0: #Returns the total number of elements in the input tensor.
        i = order[0]
        keep.append(i)

        if order.numel() == 1:
            break

        if mode == 'union':
            ovr = torch.Tensor([polygon_iou(bboxes_np[i,:], bboxes_np[j,:]) for j in order[1:]])  # heavy?
        else:
            raise TypeError('Unknown nms mode: %s.' % mode)

        ids = (ovr<=threshold).nonzero().squeeze()
        if ids.numel() == 0:
            break
        order = order[ids+1] # add 1 because of i = order[0] already use order[0] and j begin at 1
    return torch.LongTensor(keep)

if __name__ == "__main__":
    default_boxes = torch.Tensor([[0.5,0.5,1,1],[1,0.5,1,1]])
    default_boxes = change_box_order(default_boxes, 'xywh2xyxyxyxy')
    default_boxes2 = torch.Tensor([[0.5,0.5,1,1],[1,0.5,2,2]])
    default_boxes2 = change_box_order(default_boxes2, 'xywh2xyxyxyxy')
    
    print(default_boxes)
    print(polygon_area_from_tensor(default_boxes[0,:]))
    print('val')
    print(polygon_iou(default_boxes[0,:], default_boxes2[1,:]))
    print(box_nms(default_boxes, torch.Tensor([1,0.5]), threshold=0.4))
    
   
    print(quadrilateral_iou(default_boxes, default_boxes2))
    # 1.0000  0.2500
    # 0.3333  0.2500