common_params = {
    'image_size': 448,
    'batch_size': 48,
    'output_size': 7,
    'num_steps': 450000,
    'boxes_per_cell': 2,
    'num_classes': 20,
    'object_scale': 1.0,
    'noobject_scale': 0.5,
    'class_scale': 1.0,
    'coord_scale': 5.0,
}

dataset_params = {
    'train_file': '../data/train.txt',
    'test_file': '../data/test.txt'
}

path_params = {
    'checkpoints': '../checkpoints/'
}

class_names = (
    'Aeroplane', 'Bicycle', 'Bird', 'Boat', 'Bottle', 
    'Bus', 'Car', 'Cat', 'Chair', 'Cow', 'Diningtable', 
    'Dog', 'Horse', 'Motorbike', 'Person', 'Pottedplant', 
    'Sheep', 'Sofa', 'Train', 'TVmonitor'
)
