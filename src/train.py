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

        images, labels = dataset.batch()
        
        start_time = time.time()
        # Gradient tape
        with tf.GradientTape() as tape:
            predictions = model.call(images)
            loss = model.loss(predictions, labels)
        
        # Apply gradient
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        elapsed_time    = time.time() - start_time                          # Get batch time
        image_time        = elapsed_time / cfg.common_params['batch_size']
        image_speed       = 1 / image_time
        
        loss_sum += loss
        loss_avg = loss_sum / (i+1)
        if i % 10 == 0:
            print("Batch ", dataset.record_point, "/", dataset.num_batch_per_epoch, " | avg_loss ", float(loss_avg),
                " | ", round(image_speed, 3), "images/sec")

    pass
