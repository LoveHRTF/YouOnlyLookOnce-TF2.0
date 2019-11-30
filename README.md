# YouOnlyLookOnce-TF2.0
Pure Tensorflow 2.0 Keras implementation for the original [YOLO](https://arxiv.org/abs/1506.02640) Paper

This is the final project for CSCI-1470 Deep Learning @ Brown University

Currently under construction 

## Main File Structure
YouOnlyLookOnce-TF2.0
  |--src-|
         |-config.py 
         |-main.py
         |-model.py
         |-dataset.py
         |-train.py
         |-test.py
         \-visualize.py
  |-data-|
         |-voc_label.py
         |-train.txt
         |-test.txt
         \-VOCdevkit-|
                     |-VOC2007
                     \-VOC2012
  |-tool-|
         \-preprocess_pascal_voc.py

  
         

## Data Gather and Pre-process (PASCAL VOC 2012 and 2007)

### Data Download and Extraction
* cd to /YouOnlyLookOnce-TF2.0/data/, and follow the instructions [here](https://github.com/LoveHRTF/YouOnlyLookOnce-TF2.0/blob/master/data/README.md) to download and extract the PASCAL VOC 2012 and 2007 datasets

### Re-scale and Coordinate Transformation
* cd to /YouOnlyLookOnce-TF2.0/tool/, and follow the instructions [here](https://github.com/LoveHRTF/YouOnlyLookOnce-TF2.0/blob/master/tool/README.md). This step will generate path for training data `train.txt` and `test.txt`

## Train

* cd to /YouOnlyLookOnce-TF2.0/src/

* Run `python main.py`, model checkpoints will be stored in /YouOnlyLookOnce-TF2.0/checkpoints/

## Test

* TBD

## Visualization

* TBD
