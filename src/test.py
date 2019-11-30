"""
Test Function

Param inputs: Model object and dataset object
Return      : Average testing loss
"""
import time
import config as cfg

def test(model, dataset):

    loss_sum = 0
    loss_avg = 0
    

    for i in range(dataset.num_batch_per_epoch):
        start_time = time.time()
        images, labels = dataset.batch()            # Access images and labels from dataset object
        
        predictions     = model.call(images)            # Forward pass
        loss            = model.loss(predictions, labels)

        elapsed_time    = time.time() - start_time      # Get batch time
        avg_time        = elapsed_time / cfg.common_params['batch_size']
        avg_speed       = 1 / avg_time

        loss_sum += loss
        loss_avg = loss_sum / (i+1)
        
        if i % 15 == 0:
            print("Test Batch ", dataset.record_point, "/", dataset.num_batch_per_epoch, " | avg_loss ", float(loss_avg),
                    " | Speed ", avg_speed, "images/sec")


    return loss_avg