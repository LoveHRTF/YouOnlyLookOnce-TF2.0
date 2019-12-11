"""
Description: Main function for training and testing the model
"""

# Packages
import argparse
import tensorflow as tf
import os
import numpy as np
import cv2
import datetime

# Class and functions
from eval import eval
from train import train
from test import test
from dataset import Dataset
from model import Model
import visualize
from visualize import visualization, generate_prediction

# Configration file
import config as cfg

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

# Input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--num-epochs', type=int, default=400)
parser.add_argument('--mode', type=str, default='train', help='Can be "train" or "test" or "visualize" or "eval" ')
parser.add_argument('--restore', action='store_true',
                    help='Use this flag if you want to resuming training from the latest-saved checkpoint')
parser.add_argument('--visualize-number', type=int, default=128, help='Number of images generate when in visualize mode')
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

        # Tensorboard for testing
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)

        # Train Loop
        train_data = Dataset(cfg.common_params, cfg.dataset_params['train_file'])       # Training Data Preprocess 
        for epoch in range(args.num_epochs):                                            # Train
            print("============ Epoch ",epoch, "============")
            train(model, train_data, train_summary_writer, epoch)

            if epoch % 2 == 0:                                                          # Save checkpoint every other epoch
                manager.save()
                folder_name = 'epoch_' + str(epoch) + '/'
                generate_prediction(model, cfg.dataset_params['test_file'], args.visualize_number, folder_name)


    elif args.mode == 'test':
        test_data = Dataset(cfg.common_params, cfg.dataset_params['test_file'])         # Testing Data Preprocess 
        print("============ Start Testing ============")                                # Test
        test_loss = test(model, test_data)
        print("Avg_test_loss: ", float(test_loss))

    elif args.mode == 'eval':
        imageset = ('VOC2007', 'test')
        eval(imageset)
    
    # Visualization, 
    elif args.mode == 'visualize':
        generate_prediction(model, cfg.dataset_params['test_file'], args.visualize_number)


if __name__ == "__main__":
    main()
