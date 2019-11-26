import config as cfg
import tensorflow as tf

def train(model, dataset):
    '''
    Train model for one epoch. 
    '''
    
    for i in range(dataset.num_batch_per_epoch):

        images, labels = dataset.batch()
        
        with tf.GradientTape() as tape:
            predictions = model.call(images)
            loss = model.loss(predictions, labels)
        
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        if i % 1 == 0:
            print("Batch ", dataset.record_point, "/", dataset.num_batch_per_epoch, " | Loss ", loss)

    
    pass
