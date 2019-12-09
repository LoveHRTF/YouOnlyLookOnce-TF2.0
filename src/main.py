"""
Description: Main function for training and testing the model
"""

# Packages
import argparse
import tensorflow as tf
import os
import numpy as np
import cv2

# Class and functions
from train import train
from test import test
from dataset import Dataset
from model import Model
import visualize
from visualize import visualization
# Configration file
import config as cfg

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

# Input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--num-epochs', type=int, default=400)
parser.add_argument('--mode', type=str, default='train', help='Can be "train" or "test" or "visualize"')
parser.add_argument('--restore', action='store_true',
                    help='Use this flag if you want to resuming training from the latest-saved checkpoint')
parser.add_argument('--visualize-number', type=int, default=1, help='Number of images generate when in visualize mode')
args = parser.parse_args()

def main():

    # Load model
    model = Model()                                                                     # Create new model
    checkpoint = tf.train.Checkpoint(model=model)
    manager = tf.train.CheckpointManager(checkpoint, cfg.path_params['checkpoints'], max_to_keep=20)

    # Load checkpoint
    if args.restore or args.mode == 'test' or args.mode=='visualize':
        checkpoint.restore(manager.latest_checkpoint)                                   # restores the latest checkpoint
        print("Load checkpoint : ", manager.latest_checkpoint)

    # Train and test
    if args.mode == 'train':
        train_data = Dataset(cfg.common_params, cfg.dataset_params['train_file'])       # Training Data Preprocess 
        for epoch in range(args.num_epochs):                                            # Train
            print("============ Epoch ",epoch, "============")
            train(model, train_data)
            if epoch % 20 == 0:                                                         # Save checkpoint
                manager.save()

    elif args.mode == 'test':
        test_data = Dataset(cfg.common_params, cfg.dataset_params['test_file'])        # Testing Data Preprocess 
        print("============ Start Testing ============")                                # Test
        test_loss = test(model, test_data)
        print("Avg_test_loss: ", float(test_loss))
    
    # Visualization, 
    elif args.mode == 'visualize':
        print("============ Generate Visualizations ============") 
        fs_input = open(cfg.dataset_params['test_file'], 'r')
        count = 0
        for line in fs_input.readlines():
            line = line.strip().split(' ')
            # print(line[0])
            visualization(model, line[0], is_path=True, is_store=True)

            # Store certain number of images for visualization
            count += 1
            if count == args.visualize_number:
                break


if __name__ == "__main__":
    main()
