import tensorflow as tf
import cv2
import os
from model import Model
from visualize import visualization
import config as cfg

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

def main():

    # Read model from checkpoint
    model = Model()
    checkpoint = tf.train.Checkpoint(model=model)
    manager = tf.train.CheckpointManager(checkpoint, cfg.path_params['checkpoints'], max_to_keep=20)
    checkpoint.restore(manager.latest_checkpoint)   

    # Open webcam
    cam = cv2.VideoCapture(0)

    while True:

        # Read image from webcam
        _, img = cam.read()
        img = cv2.flip(img, 1)

        # Perform Forward Pass and prediction
        img_bb = visualization(model, img, is_path=False, is_store=False)

        # Update screen
        cv2.imshow('my webcam', img_bb)
        if cv2.waitKey(1) == 10: 
            break               # esc to quit

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()