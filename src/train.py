import config as cfg
import tensorflow as tf
import dataset

def train(self, model, dataset):
    '''
    Train model for one epoch. 
    '''
    batch_size = cfg.common_params['batch_size']
    
    for i in range(dataset.num_batch_per_epoch):
        images, labels, record_point, num_batch_per_epoch = dataset.batch()
        
        with tf.GradientTape() as tape:
            predictions = model.call(images)
            loss = model.loss(predictions, labels)
        
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    
    pass
