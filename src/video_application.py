"""
Description: A simple application for realtime objection using trained model
                webcam was required for running this script
"""

# Packages
import argparse
import tensorflow as tf
import cv2
import os
import numpy as np

# Class and functions
from model import Model
from visualize import visualization

# Configration file
import config as cfg

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

# Input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--mirror', type=bool, default=False)
parser.add_argument('--full-screen', type=bool, default=False)
parser.add_argument('--crop', type=bool, default=False)

args = parser.parse_args()

def main():

    # Read model from checkpoint
    model = Model(is_train=False)
    checkpoint = tf.train.Checkpoint(model=model)
    manager = tf.train.CheckpointManager(checkpoint, cfg.path_params['checkpoints'], max_to_keep=20)
    checkpoint.restore(manager.latest_checkpoint)   

    # Open webcam
    cam = cv2.VideoCapture(args.device)

    if args.full_screen:
        cv2.namedWindow("YoloV1.0", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("YoloV1.0",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

    while True:
        # Read image from webcam
        _, img = cam.read()

        if args.crop:
            shape = np.shape(img)
            y = shape[0]
            x = int((shape[1] - y) //2 )
            img = img[:, x:x+y]
            
        if args.mirror:
            img = cv2.flip(img, 1)

        # Perform Forward Pass and prediction
        img_bb = visualization(model, img, is_path=False, is_store=False)

        # Update screen
        cv2.imshow('YoloV1.0', img_bb)
        if cv2.waitKey(1) == 10: 
            break               # esc to quit

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()