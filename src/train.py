"""
Train Function

Param inputs: Model object and dataset object
"""

import config as cfg
import tensorflow as tf
import time
import datetime

def train(model, dataset, train_summary_writer, epoch):
    '''
    Train model for one epoch. 
    :param -model
    :param -dataset
    :param -train_summary_writer: tensorboard writer
    :param -epoch: current number of epoch for tensorboard log
    '''
    
    loss_sum = 0
    loss_avg = 0

    for i in range(dataset.num_batch_per_epoch):
        start_time = time.time()
        images, labels, _ = dataset.batch()
        
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

        avg_loss_str = str(float(loss_avg))
        batch_loss_str = str(float(loss))
        speed_str = str(image_speed)

        if i % 50 == 0:
            print("Batch ", dataset.record_point-1, "/", dataset.num_batch_per_epoch, 
                " | avg_loss ", avg_loss_str[0:8],
                " | batch_loss", batch_loss_str[0:8], 
                " | train_speed", speed_str[0:5], "images/sec")

            with train_summary_writer.as_default():
                tf.summary.scalar('train_loss', loss, step=i * (epoch + 1))

    return loss_avg
