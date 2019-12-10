import cv2
import os
import time

import config as cfg
from visualize import decoder


VOC_CLASSES = cfg.class_names
eval_dir = '../evaluation'
result_dir = os.path.join(eval_dir, 'result')

if not os.path.exists(eval_dir):
     os.mkdir(eval_dir)

if not os.path.exists(result_dir):
    os.mkdir(result_dir)


def test(model, dataset, log_result=True):
    """Test Function

    @param model: Model object
    @param dataset: Dataset object
    @param log_result: set true if you need to write testing result of each class to disk and do evaluation
    @return: Average testing loss
    """

    loss_sum = 0
    loss_avg = 0
    if log_result:
        det_list = [os.path.join(result_dir, file) for file in os.listdir(result_dir)]
        for det_class_file in det_list:
            with open(det_class_file, mode='w') as f:
                pass
    
    for i in range(dataset.num_batch_per_epoch):
        start_time = time.time()
        images, labels, fnames = dataset.batch()                        # Access images and labels from dataset object

        predictions     = model(images)                         # Forward pass
        loss            = model.loss(predictions, labels)

        elapsed_time    = time.time() - start_time              # Get batch time
        image_speed     = 1 / (elapsed_time / cfg.common_params['batch_size'])

        loss_sum += loss
        loss_avg = loss_sum / (i+1)
        
        loss_str = str(float(loss_avg))
        speed_str = str(image_speed)
        if i % 15 == 0:
            print("Test Batch ", dataset.record_point-1, "/", dataset.num_batch_per_epoch, 
                " | avg_loss ", loss_str[0:8], " | test_speed ", speed_str[0:5], "images/sec")

    if log_result:
        for i, fname in enumerate(fnames):
            img = cv2.imread(fname)
            h, w, _ = img.shape
            boxes, cls_indexs, probs = decoder(predictions[i])

            for j, box in enumerate(boxes):
                x1 = int(box[0] * w)
                y1 = int(box[1] * h)
                x2 = int(box[2] * w)
                y2 = int(box[3] * h)
                cls_index = int(cls_indexs[j])
                prob = probs[j]
                class_name = VOC_CLASSES[cls_index]

                image_id = fname.split('/')[-1].split('.')[0]
                filename = os.path.join(result_dir, class_name + '.txt')
                with open(filename, mode='a') as file:
                    content = image_id + ' ' + str(prob) + ' ' + str(int(x1)) + ' ' + str(int(y1)) + ' ' + str(
                        int(x2)) + ' ' + str(int(y2)) + '\n'
                    file.write(content)

    return loss_avg
