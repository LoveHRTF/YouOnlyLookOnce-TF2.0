# Utilities

To get the PASCAL VOC data and preprocess it

1. Download the data to directory `~/data`
```
wget https://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar
wget https://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar
wget https://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar
tar xf VOCtrainval_11-May-2012.tar
tar xf VOCtrainval_06-Nov-2007.tar
tar xf VOCtest_06-Nov-2007.tar 
```

2. Run *preprocess_pascal_voc.py* to extract labels and bounding boxes, which will be stored in 
`YouOnlyLookOnce-TF2.0/data/`
