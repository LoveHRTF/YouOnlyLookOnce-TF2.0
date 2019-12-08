import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import config as cfg


def visualization(model, img_path):
    """
    Visualize an image with object detextion logitsictions.
    :param model: Model object
    :param img_path: Image path for testing
    
    :return: New image with bounding boxes and class names.
    """
    # bgr_mean = (103.939, 116.779, 123.68)  # bgr
    # img = cv2.imread(img_path)
    img = img_path

    # resize origin image
    image_size = cfg.common_params['image_size']
    img = cv2.resize(img, (image_size, image_size))
    img = tf.reshape(img, (1, image_size, image_size, 3))
    # img = img - np.array(bgr_mean, dtype=np.float32)

    logits = model(img)
    boxes, class_idx, scores = decoder(logits, conf_thresh=0.1, score_thresh=0.1)

    for i in range(boxes.shape[0]):
        x1 = boxes[i, 0]
        y1 = boxes[i, 1]
        x2 = boxes[i, 2]
        y2 = boxes[i, 3]
        
        # draw a green rectangle to visualize the bounding box
        image = cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)

        # print class name (a green text)
        class_name = cfg.class_names[class_idx[i]]
        image = cv2.putText( 
            image, class_name, (x1, y1), font=cv2.FONT_HERSHEY_SIMPLEX, 
            fontScale=1, color=(0,255,0), thickness=2, lineType=cv2.LINE_AA )
        
    cv2.imshow('show_result',image)

    cv2.imwrite('show_result.jpg', image)


def decoder(logits, conf_thresh=0.1, score_thresh=0.1):
    """Decode the output of model
    :param logits: output of the model, size 7 x 7 x 30
    :param conf_thresh: threshold of confidence, above which indicates an object in the cell
    :param score_thresh: threshold of class-specific confidence score

    :return: boxes(x1, y1, x2, y2), cls_indices, scores
    """

    boxes = []
    cls_indices = []
    scores = []

    grid_num = cfg.common_params['output_size']
    cell_size = 1./grid_num
    logits = np.squeeze(logits)

    # Compute mask for picking bbx in each cell
    conf1 = logits[:, :, 4][:, :, np.newaxis]
    conf2 = logits[:, :, 9][:, :, np.newaxis]
    conf = np.concatenate((conf1, conf2),axis=-1)
    mask = (conf >= conf_thresh) + (conf == conf.max())

    for i in range(grid_num):
        for j in range(grid_num):
            for b in range(2):
                if mask[i, j, b]:
                    # Get x, y, w, h, confidence of bbx,
                    # (x, y) represents the center of the bbx relative to the bounds of the grid cell
                    # w, h are predicted relative to the whole image.
                    box = logits[i, j, b * 5: b * 5 + 4].copy()
                    confidence = logits[i, j, b * 5 + 4].copy()

                    # Compute the offset of the cell
                    # Convert the center of bbx to image coordinates system
                    offset = np.array([j, i]) * cell_size
                    box[:2] = box[:2] * cell_size + offset
                    box_xy = np.zeros_like(box)

                    # Compute the upper left and bottom right coordinates of bbx
                    box_xy[:2] = box[:2] - 0.5 * box[2:]
                    box_xy[2:] = box[:2] + 0.5 * box[2:]

                    # Get max prob and corresponding class index
                    max_prob = np.max(logits[i, j, 10:])
                    cls_index = np.argmax(logits[i, j, 10:])

                    if confidence * max_prob > score_thresh:
                        boxes.append(box_xy)
                        cls_indices.append(cls_index)
                        scores.append(confidence * max_prob)

    if len(boxes) == 0:
        boxes = np.zeros((1,4))
        scores = np.zeros(1)
        cls_indices = np.zeros(1)
    else:
        boxes = np.array(boxes)
        scores = np.array(scores)
        cls_indices = np.array(cls_indices)
    keep = nms(boxes, scores)

    return boxes[keep], cls_indices[keep], scores[keep]


def nms(boxes, scores, overlap_thresh=0.5):
    """Non-maximum suppression
    :param boxes: bounding boxes holding (x1, y1, x2, y2)
    :param scores: class-specific score of each bbx
    :param overlap_thresh: threshold of overlap

    :return: the indices of bbx to keep
    """

    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    areas = (x2 - x1) * (y2 - y1)
    order = np.argsort(scores)[::-1]
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break

        xx1 = x1[order[1:]].clip(min=x1[i])
        yy1 = y1[order[1:]].clip(min=y1[i])
        xx2 = x2[order[1:]].clip(max=x2[i])
        yy2 = y2[order[1:]].clip(max=y2[i])

        w = (xx2-xx1).clip(min=0)
        h = (yy2-yy1).clip(min=0)
        inter = w*h

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        ids = (ovr <= overlap_thresh).nonzero()[0]
        if len(ids) == 0:
            break
        order = order[ids+1]
    return keep
