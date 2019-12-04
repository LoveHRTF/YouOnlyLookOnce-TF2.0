# Packages
import argparse
import tensorflow as tf
# Class and functions
from train import train
from test import test
from dataset import Dataset
from model import Model
from visualize import visualize
# Configration file
import config as cfg

# Input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--num-epochs', type=int, default=400)
parser.add_argument('--mode', type=str, default='train', help='Can be "train" or "test"')
parser.add_argument('--restore', action='store_true',
                    help='Use this flag if you want to resuming training from the latest-saved checkpoint')
args = parser.parse_args()

def main():

    # Load model
    model = Model()                                                                     # Create new model
    checkpoint = tf.train.Checkpoint(model=model)
    manager = tf.train.CheckpointManager(checkpoint, cfg.path_params['checkpoints'], max_to_keep=20)

    # Load checkpoint
    if args.restore or args.mode == 'test':
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

    # Visualization


if __name__ == "__main__":
    main()
