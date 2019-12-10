import cv2
import numpy as np
import tensorflow as tf
import os
import config as cfg


def generate_prediction(model, image_list, image_number, storage_folder='default_folder/'):
    """
    Visualize a list of image, draw bounding boxes, and store in local folder
    : param model: Model object
    : param image_list: txt list of images
    : param image_number: number of images to generate
    : param storage_folder: subfolder directory under ~/tmp 

    : return: None
    """
    print("============ Generate Visualizations ============")
    count = 0
    fs_input = open(image_list, 'r')
    for line in fs_input.readlines():
        line = line.strip().split(' ')
        print(count, '/', image_number, ':', line[0])
        visualization(model, line[0], is_path=True, is_store=True, storage_folder=storage_folder)
        fs_input.close()
        # Store certain number of images for visualization
        count += 1
        if count >= image_number:
            break
    print('===== Result generation finished, stored in ~/tmp/' + storage_folder + ' =====')


def visualization(model, img, is_path=True, is_store=True, storage_folder='default_folder/'):
    """
    Visualize an image with object detection logitsictions.
    :param model: Model object
    :param img_path: Image path for testing
    
    :return: New image with bounding boxes and class names.
    """
    if is_path:
        img_ = cv2.imread(img)
        # Get image name
        img_name = img.split('/')
        img_name = img_name[-1]
    else:
        img_ = img

    h, w, _ = img_.shape
    # resize origin image
    image_size = int(cfg.common_params['image_size'])
    img = cv2.resize(img_, (image_size, image_size))

    # Forward Pass, prediction
    model_input = tf.reshape(img, (-1, image_size, image_size, 3))
    model_input = tf.dtypes.cast(model_input, tf.float32)
    logits = model(model_input)
    boxes, class_idx, scores = decoder(logits, conf_thresh=0.15, score_thresh=0.15)

    if len(boxes) > 0:
        for i in range(boxes.shape[0]):
            x1 = int(boxes[i, 0] * w)
            y1 = int(boxes[i, 1] * h)
            x2 = int(boxes[i, 2] * w)
            y2 = int(boxes[i, 3] * h)
            
            # draw a green rectangle to visualize the bounding box
            # start_point = (x1+30 * i, y1+30 * i)
            start_point = (x1, y1)
            end_point = (x2, y2)
            color = (0, 255, 0)
            thickness = 2
            fontScale = 1

            image = cv2.rectangle(img_, start_point, end_point, color, thickness)
            # print class name (a green text)
            class_name = cfg.class_names[int(class_idx[i])]
            score = str(scores[i])
            title = class_name + ':' + score[0:6]
            image = cv2.putText( 
                image, title, start_point, cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale, color, thickness, cv2.LINE_AA)
                
    else:
        image = img_

    if is_store:
        # Generate Path
        path = '../tmp/' + storage_folder
        if not os.path.exists(path):
            os.makedirs(path)
        # Store image
        cv2.imwrite(path + img_name, image)

    return image


def decoder(logits, conf_thresh=0.1, score_thresh=0.2):
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


def nms(boxes, scores, overlap_thresh=0.25, score_thresh=0.25):
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
        if scores[i] > score_thresh:
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