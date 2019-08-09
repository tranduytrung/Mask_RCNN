import numpy as np
import cv2 as cv
import skimage.draw as draw

def bytescaling(data, cmin=None, cmax=None, high=255, low=0):
    """
    Converting the input image to uint8 dtype and scaling
    the range to ``(low, high)`` (default 0-255). If the input image already has 
    dtype uint8, no scaling is done.
    :param data: 16-bit image data array
    :param cmin: bias scaling of small values (def: data.min())
    :param cmax: bias scaling of large values (def: data.max())
    :param high: scale max value to high. (def: 255)
    :param low: scale min value to low. (def: 0)
    :return: 8-bit image data array
    """
    if data.dtype == np.uint8:
        return data

    if high > 255:
        high = 255
    if low < 0:
        low = 0
    if high < low:
        raise ValueError("`high` should be greater than or equal to `low`.")

    if cmin is None:
        cmin = data.min()
    if cmax is None:
        cmax = data.max()

    cscale = cmax - cmin
    if cscale == 0:
        cscale = 1

    scale = float(high - low) / cscale
    bytedata = (data - cmin) * scale + low
    return (bytedata.clip(low, high) + 0.5).astype(np.uint8)


def normalize_minmax(frame, min_value, max_value):
    return (frame - min_value) / (max_value - min_value)


def clip_and_fill(frame, min_value, max_value, fill_value="uniform"):
    nan_mask = np.isnan(frame)
    if fill_value == "uniform":
        fill_value = np.random.uniform(
            min_value, max_value, size=np.sum(nan_mask))
    elif fill_value == "normal":
        mean = (min_value + max_value) / 2
        std = (max_value - mean) / 4  # since 2 std = 98% of coverage
        fill_value = np.random.normal(
            mean, std, size=np.sum(nan_mask))

    frame[nan_mask] = fill_value
    clipped = np.clip(frame, min_value, max_value)
    return clipped

def resize_to(image, dsize):
    if image.shape[0] != dsize[0] or image.shape[1] != dsize[1]:
        return cv.resize(image, dsize[::-1])
    return image



def compute_iou(box, boxes, box_area, boxes_area):
    """Calculates IoU of the given box with the array of the given boxes.
    box: 1D vector [y1, x1, y2, x2]
    boxes: [boxes_count, (y1, x1, y2, x2)]
    box_area: float. the area of 'box'
    boxes_area: array of length boxes_count.

    Note: the areas are passed in rather than calculated here for
    efficiency. Calculate once in the caller to avoid duplicate work.
    """
    # Calculate intersection areas
    y1 = np.maximum(box[0], boxes[:, 0])
    y2 = np.minimum(box[2], boxes[:, 2])
    x1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    union = box_area + boxes_area[:] - intersection[:]
    iou = intersection / union
    return iou


def compute_overlaps(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2)].

    For better performance, pass the largest set first and the smaller second.
    """
    # Areas of anchors and GT boxes
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Compute overlaps to generate matrix [boxes1 count, boxes2 count]
    # Each cell contains the IoU value.
    overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))
    for i in range(overlaps.shape[1]):
        box2 = boxes2[i]
        overlaps[:, i] = compute_iou(box2, boxes1, area2[i], area1)
    return overlaps

def compute_matchesv2(pred_boxes, gt_boxes, query_iou=0.5):
    """ compute the matches given that the boxes are same class
    Input:
        pred_boxes [n_pred, (y1,x1,y2,x2)]
        gt_boxes [n_gt, (y1,x1,y2,x2)]
        iou_threshold float
    """
    pred_match = np.zeros([pred_boxes.shape[0]], dtype=np.bool)
    if pred_match.shape[0] == 0 or gt_boxes.shape[0] == 0:
        return pred_match

    # Compute IoU overlaps [pred_masks, gt_masks]
    overlaps = compute_overlaps(pred_boxes, gt_boxes)
    darg = np.argsort(-overlaps, axis=1) # descending
    max_gt_indices = darg[:, 0]
    gt_status = - np.ones([gt_boxes.shape[0]], dtype=np.int32)
    for i in range(pred_match.shape[0]):
        gt_index = max_gt_indices[i]
        iou = overlaps[gt_index]
        if iou < query_iou:
            continue

        if gt_status[gt_index] > -1:
            # roll back
            pred_match[gt_status[gt_index]] = False
            gt_status[gt_index] = -2
        elif gt_status[gt_index] == -1:
            gt_status[gt_index] = i
            pred_match[i] = True
        
    return pred_match

def compute_apv2(pred_boxes, pred_ids, pred_scores, gt_boxes, gt_ids, query_id, query_iou=0.5):
    """ compute AP of each class_ids given a set of predicted boxes and groundtruth boxes
        each item in a list is a corresponding image
    Inputs:
        pred_boxes list([n_pred, (y1,x1,y2,x2)])
        pred_ids list([n_pred])
        pred_scores list([n_pred])
        gt_boxes list([n_gt, (y1,x1,y2,x2)])
        gt_ids list([n_gt])
        query_id (int)
        iou_threshold (float)
    """

    assert len(pred_boxes) == len(pred_ids) == len(gt_boxes) == len(gt_ids)
    n_instances = len(pred_boxes)

    pred_matches = []
    pred_scores2 = []
    n_gt = 0

    for i in range(n_instances):
        # filter the query id        
        pred_qmask = pred_ids[i] == query_id
        cpred_boxes = pred_boxes[i][pred_qmask]
        cpred_scores = pred_scores[i][pred_qmask]

        gt_qmask = gt_ids[i] == query_id
        cgt_boxes = gt_boxes[i][gt_qmask]

        # compute maches
        cpred_matches = compute_matchesv2(cpred_boxes, cgt_boxes, query_iou)

        # update
        pred_matches.append(cpred_matches)
        n_gt += cgt_boxes.shape[0]
        pred_scores2.append(cpred_scores)

    # flatten
    pred_matches = np.concatenate(pred_matches)
    pred_scores2 = np.concatenate(pred_scores2)

    # sort by score
    sarg = np.argsort(pred_scores2)[::-1]
    pred_matches = pred_matches[sarg]

    # compute precision
    precisions = np.cumsum(pred_matches) / (np.arange(len(pred_matches)) + 1)
    recalls = np.cumsum(pred_matches) / n_gt

    # Pad with start and end values to simplify the math
    precisions = np.concatenate([[0], precisions, [0]])
    recalls = np.concatenate([[0], recalls, [1]])

    # Ensure precision values decrease but don't increase. This way, the
    # precision value at each recall threshold is the maximum it can be
    # for all following recall thresholds, as specified by the VOC paper.
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = np.maximum(precisions[i], precisions[i + 1])

    # Compute mean AP over recall range
    indices = np.where(recalls[:-1] != recalls[1:])[0] + 1
    AP = np.sum((recalls[indices] - recalls[indices - 1]) *
                 precisions[indices])
    
    return AP

def compute_mAP(pred_boxes, pred_ids, pred_scores, gt_boxes, gt_ids, 
    query_ids=None, query_ious=np.arange(0.5, 1.0, 0.05)):

    if query_ids is None:
        query_ids = np.unique(np.concatenate(gt_ids))

    APs = []
    for iou_threshold in query_ious:
        for cat in query_ids:
            APs.append(compute_apv2(pred_boxes, pred_ids, pred_scores, gt_boxes, gt_ids, query_id=cat, query_iou=iou_threshold))

    return np.mean(APs)

def bboxes2masks(bboxes, shape):
    masks = np.zeros((*shape, len(bboxes)), dtype=np.uint8)
    for i, bbox in enumerate(bboxes):
        # each bbox is y1, x1, y2, x2
        y1, x1, y2, x2 = bbox
        start = (int(y1), int(x1))
        end = (int(y2), int(x2))
        rr, cc = draw.rectangle(start, end=end, shape=shape)
        masks[rr, cc, i] = 255

    return masks

def resize_bbox(np_bbox, src_size, dst_size):
    """ resize the bounding box from src image size to destinate image size

    np_bbox: [# of boxes, (y1, x1, y2, x2)]
    src_size: (height, width)
    dst_size: (height, width)
    """
    ratio = np.array(dst_size, dtype=np.float32) / src_size
    y1 = np_bbox[:, 0] * ratio[0]
    x1 = np_bbox[:, 1] * ratio[1]
    y2 = np_bbox[:, 2] * ratio[0]
    x2 = np_bbox[:, 3] * ratio[1]
    return np.stack((y1, x1, y2, x2), axis=-1)

def normalize_rgb(data):
    """ normalize rgb
    data [H, W, C], data type is float32 (0 to 1) or uint8 (0 to 255)
    """

    if data.dtype == np.float32:
        return (data - 0.5) / 0.5
    else:
        return (data.astype(np.float32) - 127.0) / 127.0

def normalize_d(data):
    """ normalize depth data
    data [H, W, C], data type is float32 (meter)
    """
    return (np.clip(data, 0.2, 1.2) - 0.7) / 0.5

