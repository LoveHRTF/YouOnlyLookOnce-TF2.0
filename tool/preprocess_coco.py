#-*-coding:utf-8-*-

import os
from xml.etree.ElementTree import dump
import json
import pprint
import argparse
from coco_format import COCO, YOLO

def main():

    # Generate test.txt
    coco = COCO()
    yolo = YOLO(os.path.abspath("../data/coco/coco.names"))
    flag, data = coco.parse("../data/coco/annotations/instances_val2014.json")

    if flag == True:

        flag, data = yolo.save(data, "../data/coco/images/train2014/", ".jpg", "../data/", "test_coco.txt")

    else:
        print("COCO Testing Data Parsing Result : {}, msg : {}".format(flag, data))

    # Generate train.txt
    coco = COCO()
    yolo = YOLO(os.path.abspath("../data/coco/coco.names"))
    flag, data = coco.parse("../data/coco/annotations/instances_train2014.json")

    if flag == True:

        flag, data = yolo.save(data, "../data/coco/images/train2014/", ".jpg", "../data/", "train_coco.txt")

    else:
        print("COCO Training Data Parsing Result : {}, msg : {}".format(flag, data))

if __name__ == '__main__':

    main()
