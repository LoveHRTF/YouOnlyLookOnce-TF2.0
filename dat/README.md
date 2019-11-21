# Training data and label files

### Using Pascal VOC Data

#### Download Dataset
`wget https://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar`

`wget https://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar`

`wget https://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar`

#### Unzip
`tar xf VOCtrainval_11-May-2012.tar`
`tar xf VOCtrainval_06-Nov-2007.tar`
`tar xf VOCtest_06-Nov-2007.tar`

#### Generate Path and Labels
`wget https://pjreddie.com/media/files/voc_label.py`
`python voc_label.py`

#### Cat all files into one list
`cat 2007_* 2012_train.txt > train.txt`

* Instructions from [here](https://pjreddie.com/darknet/yolov1/)
