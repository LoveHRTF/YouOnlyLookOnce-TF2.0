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
    #TODO: Main function
    # Preprocess 
    dataset = Dataset(cfg.common_params, cfg.dataset_params['train_file'])
    #
    model = Model()

    # Train
    for epoch in range(150):
        print("============ Epoch ", epoch, "============")
        train(model, dataset)


    # Test




if __name__ == "__main__":
    main()

