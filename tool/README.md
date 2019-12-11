# Utilities


## PASCAL-VOC Dataset
To get the PASCAL VOC data and preprocess:

1. Download the data to directory `YouOnlyLookOnce-TF2.0/data` by following instruction in it

2. Run
`python preprocess_pascal_voc.py`
to extract labels and bounding boxes, which will be stored in `YouOnlyLookOnce-TF2.0/data/`

## Coco 2017
To get coco data and preprocess:

1. Build file directory
* mkdir `YouOnlyLookOnce-TF2.0/data/coco/`
* mkdir `YouOnlyLookOnce-TF2.0/data/coco/annotations`
* mkdir `YouOnlyLookOnce-TF2.0/data/coco/images`

2. File Download

The download link for Coco 2017 dataset can be found [here](http://cocodataset.org/#download) /

* Download coco **2017 Train images**, and unzip in `YouOnlyLookOnce-TF2.0/data/coco/images`
* Download coco **2017 Val images**, and unzip in `YouOnlyLookOnce-TF2.0/data/coco/images`
* Download coco **2017 Train/Val annotations**, and unzip in `YouOnlyLookOnce-TF2.0/data/coco/annotations`
* Move **coco.name** to `YouOnlyLookOnce-TF2.0/data/coco/`

Now the file structure should look like this: /

```
~\YouOnlyLookOnce-TF2.0

  --data-|
         \-coco-|
                |-annotations-|
                              |-instance_train2017.json
                              \-instance_val2017.json
                |
                |-images-|
                         |-train2017/
                         |-val2017/
                |
                \-coco.names
                     
```

3. Run
`python preprocess_cooc.py`
to extract labels and bounding boxes, which will be stored in `YouOnlyLookOnce-TF2.0/data/`
