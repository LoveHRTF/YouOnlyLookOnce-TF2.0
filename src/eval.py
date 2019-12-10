import numpy as np
import os

from voc_eval import voc_eval


eval_dir = '../evaluation'
result_dir = os.path.join(eval_dir, 'result')


def eval(imageset):
    if not os.path.exists(eval_dir):
        os.mkdir(eval_dir)

    det_list = [os.path.join(result_dir, file) for file in os.listdir(result_dir)]
    det_classes = list()
    for file in det_list:
        classes = os.path.splitext(os.path.basename(file))[0].split('_')[-1]
        det_classes.append(classes)
        detpath = file.replace(classes, '%s')

    YOLO_PATH = os.path.abspath("../")
    VOC_PATH = os.path.join(YOLO_PATH, 'data', 'VOCdevkit', imageset[0])

    annopath = os.path.join(VOC_PATH, 'Annotations', '%s.xml')
    imagesetfile = os.path.join(VOC_PATH, 'ImageSets', 'Main', imageset[1] + '.txt')

    MAPList = list()
    for classname in det_classes:
        rec, prec, ap = voc_eval(detpath, annopath, imagesetfile, classname, eval_dir)
        print('%s\t AP:%.4f' % (classname, ap))
        MAPList.append(ap)

    Map = np.array(MAPList)
    mean_Map = np.mean(Map)
    print('------ Map: %.4f' % (mean_Map))
