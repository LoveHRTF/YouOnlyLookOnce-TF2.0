import config as cfg
import tensorflow as tf

def train(model, dataset):
    '''
    Train model for one epoch. 
    '''
    
    loss_sum = 0
    loss_avg = 0

    for i in range(dataset.num_batch_per_epoch):

        images, labels = dataset.batch()
        
        # Gradient tape
        with tf.GradientTape() as tape:
            predictions = model.call(images)
            loss = model.loss(predictions, labels)
        
        # Apply gradient
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        loss_sum += loss
        loss_avg = loss_sum / (i+1)
        if i % 5 == 0:
            print("Batch ", dataset.record_point, "/", dataset.num_batch_per_epoch, " | avg_loss ", float(loss_avg))

    pass
