import tensorflow as tf
import dataset, model, train, test, config

def main():
    #TODO: Main function
    # Preprocess 
    dataset = Dataset(cfg.common_params, cfg.dataset_params['train_file'])
    

    #
    model = Model()

    # Train
    for epoch in range(150):
        # adjust lr, lr = 1e-3 for first 5 epochs
        if epoch > 5:                   # next 10 epochs, slowly raise lr from 1e-3 to 1e-2
            model.learning_rate += 1e-3
        else if epoch > 15:             # next 75 epochs, lr = 1e-2
            model.learning_rate = 1e-2
        else if epoch > 90:             # next 30 epochs, lr = 1e-3
            model.learning_rate = 1e-3
        else if epoch > 120:            # next 30 epochs, lr = 1e-4
            model.learning_rate = 1e-4
        
        train(model, inputs, labels)


    # Test




if __name__ == "__main__":
    pass

