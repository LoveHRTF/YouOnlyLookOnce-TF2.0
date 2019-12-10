common_params = {
    'image_size': 448,
    'batch_size': 42,
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
    'kite',
    'baseball bat',
    'baseball glove',
    'skateboard',
    'surfboard',
    'tennis racket',
    'bottle',
    'wine glass',
    'cup',
    'fork',
    'knife',
    'spoon',
    'bowl',
    'banana',
    'apple',
    'sandwich',
    'orange',
    'broccoli',
    'carrot',
    'hot dog',
    'pizza',
    'donut',
    'cake',
    'chair',
    'couch',
    'potted plant',
    'bed',
    'dining table',
    'toilet',
    'tv',
    'laptop',
    'mouse',
    'remote',
    'keyboard',
    'cell phone',
    'microwave',
    'oven',
    'toaster',
    'sink',
    'refrigerator',
    'book',
    'clock',
    'vase',
    'scissors',
    'teddy bear',
    'hair drier',
    'toothbrush'
)


