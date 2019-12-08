"""
Train Function

Param inputs: Model object and dataset object
"""

import config as cfg
import tensorflow as tf
import time

def train(model, dataset):
    '''
    Train model for one epoch. 
    '''
    
    loss_sum = 0
    loss_avg = 0

    for i in range(dataset.num_batch_per_epoch):
        start_time = time.time()
        images, labels = dataset.batch()
        
        # Gradient tape
        with tf.GradientTape() as tape:
            predictions = model(images)
            loss = model.loss(predictions, labels)
        
        # Apply gradient
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        # Timing
        elapsed_time    = time.time() - start_time
        image_speed     = 1 / (elapsed_time / cfg.common_params['batch_size'])
        
        loss_sum += loss
        loss_avg = loss_sum / (i+1)

        loss_str = str(float(loss_avg))
        speed_str = str(image_speed)

        if i % 15 == 0:
            print("Batch ", dataset.record_point-1, "/", dataset.num_batch_per_epoch, 
                " | avg_loss ", loss_str[0:8], " | train_speed", speed_str[0:5], "images/sec")

    pass
