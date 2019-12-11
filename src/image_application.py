"""
Description: A simple application for single image objection using trained model
"""

# Packages
import argparse
import tensorflow as tf
import cv2
import os
# Class and functions
from model import Model
from visualize import visualization

# Configration file
import config as cfg

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

# Input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='../', help='input image path')
args = parser.parse_args()

def main():
    # Read model from checkpoint
    model = Model()
    checkpoint = tf.train.Checkpoint(model=model)
    manager = tf.train.CheckpointManager(checkpoint, cfg.path_params['checkpoints'], max_to_keep=20)
    checkpoint.restore(manager.latest_checkpoint)   

    # Perform Forward Pass and prediction
    _ = visualization(model, args.path, is_path=True, is_store=True, storage_folder='single_images/')

if __name__ == '__main__':
    main()