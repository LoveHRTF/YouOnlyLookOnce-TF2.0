# Packages
import tensorflow as tf

# Class and functions
# import dataset, model, train, test
from train import train
from test import test
from dataset import Dataset
from model import Model

# Configration file
import config as cfg

# """ Test """
# m = Model()
# d = Dataset(cfg.common_params, cfg.dataset_params['train_file'])
# img, label, _, _ = d.batch()
# jj = m(img)
# jb = m.loss(jj, label)
# """ End Test """

def main():
    # Preprocess 
    dataset = Dataset(cfg.common_params, cfg.dataset_params['train_file'])

    # Load model
    model = Model()             # Create new model
    checkpoint = tf.train.Checkpoint(model=model)
    manager = tf.train.CheckpointManager(checkpoint, cfg.path_params['checkpoints'], max_to_keep=10)

    # Load checkpoint
    is_restore = False
    if is_restore:
        checkpoint.restore(manager.latest_checkpoint)
        print("Load checkpoint : ", manager.latest_checkpoint)

    # Train
    for epoch in range(1):
        print("============ Epoch ", epoch, "============")
        train(model, dataset)

        # Save checkpoint
        if epoch % 10 == 0:
            manager.save()

    # TODO: Test




if __name__ == "__main__":
    main()

