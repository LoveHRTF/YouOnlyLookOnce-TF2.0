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
        images, labels = dataset.batch()                # Access images and labels from dataset object

        predictions     = model.call(images, is_test=True)              # Forward pass
        loss            = model.loss(predictions, labels)

        elapsed_time    = time.time() - start_time                      # Get batch time
        image_speed     = 1 / (elapsed_time / cfg.common_params['batch_size'])

        loss_sum += loss
        loss_avg = loss_sum / (i+1)
        
        loss_str = str(float(loss_avg))
        speed_str = str(image_speed)
        if i % 15 == 0:
            print("Test Batch ", dataset.record_point-1, "/", dataset.num_batch_per_epoch, 
                " | avg_loss ", loss_str[0:8], " | test_speed ", speed_str[0:5], "images/sec")

    return loss_avg
    