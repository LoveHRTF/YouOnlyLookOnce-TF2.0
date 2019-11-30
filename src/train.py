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
        image_speed     = 1 / (elapsed_time / cfg.common_params['batch_size'])
        
        loss_sum += loss
        loss_avg = loss_sum / (i+1)
        
        if i % 15 == 0:
            print("Batch ", dataset.record_point-1, "/", dataset.num_batch_per_epoch, " | avg_loss ", round(float(loss_avg), 6),
                " | ", round(image_speed, 3), "images/sec")

    pass
