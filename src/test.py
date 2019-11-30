"""
Test Function

Param inputs: Model object and dataset object
Return      : Average testing loss
"""

def test(model, dataset):

    loss_sum = 0
    loss_avg = 0

    for i in range(dataset.num_batch_per_epoch):

        images, labels = dataset.batch()            # Access images and labels from dataset object
        
        predictions = model.call(images)            # Forward pass
        loss = model.loss(predictions, labels)

        loss_sum += loss
        loss_avg = loss_sum / (i+1)
        
        if i % 15 == 0:
            print("Test Batch ", dataset.record_point, "/", dataset.num_batch_per_epoch, " | avg_loss ", float(loss_avg))


    return loss_avg