# YouOnlyLookOnce-TF2.0
Pure Tensorflow 2.0 Keras implementation for the original [YOLO](https://arxiv.org/abs/1506.02640) Paper

This is the final project for CSCI-1470 Deep Learning @ Brown University

Currently under construction 

## Main File Structure
### Code
```
~\YouOnlyLookOnce-TF2.0

  --src-|
        |-config.py
        |-main.py
        |-model.py
        |-dataset.py
        |-train.py
        |-test.py
        \-visualize.py
        
  --data-|
         |-voc_label.py
         |-train.txt
         |-test.txt
         \-VOCdevkit-|
                     |-VOC2007
                     \-VOC2012
                     
  --tool-|
         \-preprocess_pascal_voc.py
```

### Model and Temporary Files
```
  --checkpoints-|
                |-checkpoint
                |-ckpt-xx.index
                |-ckpt-xx.data-00000-of-00002
                \-ckpt-xx.data-00001-of-00002
  
  --tmp-|
        \- TBD
  
  --doc-|
        \- TBD
```

## Data Gather and Pre-process (PASCAL VOC 2012 and 2007)

### Data Download and Extraction
* Under ~/YouOnlyLookOnce-TF2.0/src/  \
Follow the instructions [here](https://github.com/LoveHRTF/YouOnlyLookOnce-TF2.0/blob/master/data/README.md) \
This step will download and extract the PASCAL VOC 2012 and 2007 datasets

### Re-scale and Coordinate Transformation
* Under ~/YouOnlyLookOnce-TF2.0/src/  \
Follow the instructions [here](https://github.com/LoveHRTF/YouOnlyLookOnce-TF2.0/blob/master/tool/README.md). \
This step will generate path for training data `train.txt` and `test.txt`

## Train
* Under ~/YouOnlyLookOnce-TF2.0/src/  \
### Train a new model
To train from scratch, model checkpoints will be stored in /YouOnlyLookOnce-TF2.0/checkpoints/: \
`python main.py` 
### Restore from a checkpoint
To resume training from the latest saved checkpoint: \
`python main.py restore` 

## Test
* Under ~/YouOnlyLookOnce-TF2.0/src/  \
To test the latest checkpoint on test set: \
`python main.py --mode=test`

## Visualization

* TBD
