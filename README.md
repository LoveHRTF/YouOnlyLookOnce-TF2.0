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
        |-video_application.py
        |-image-application.py
        \-visualize.py-|
                       |-generate_prediction()
                       |-visualization()
                       |-decoder()
                       \-nms()
        
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

## 1. Data Gather and Pre-process (PASCAL VOC 2012 and 2007)

### 1.1. Data Download and Extraction
Under dir ~/YouOnlyLookOnce-TF2.0/src/  \

Follow the instructions [here](https://github.com/LoveHRTF/YouOnlyLookOnce-TF2.0/blob/master/data/README.md) \
This step will download and extract the PASCAL VOC 2012 and 2007 datasets

### 1.2. Re-scale and Coordinate Transformation
Under dir ~/YouOnlyLookOnce-TF2.0/src/  \

Follow the instructions [here](https://github.com/LoveHRTF/YouOnlyLookOnce-TF2.0/blob/master/tool/README.md). \
This step will generate path for training data `train.txt` and `test.txt`

## 2. Train
Under dir ~/YouOnlyLookOnce-TF2.0/src/  \

Test outputs will be generated every time when model was auto saved (per 20 epochs), and located in ` /YouOnlyLookOnce-TF2.0/tmp/epoch_n`

### 2.1. Train a new model
To train from scratch, model checkpoints will be stored in /YouOnlyLookOnce-TF2.0/checkpoints/: \
`python main.py` 
### 2.2. Restore from a checkpoint
To resume training from the latest saved checkpoint: \
`python main.py restore` 

## 3. Test
Under dir ~/YouOnlyLookOnce-TF2.0/src/  \

To test the latest checkpoint on test set: \
`python main.py --mode=test`

## 4. Visualization

### 4.1. Real-time Video Detection
We have developed a simple script to visualize the detection, a webcam is required. \
To perform realtime detection: \

* Place trained model under `~/YouOnlyLookOnce-TF2.0/checkpoints`
* Connect a webcam and ensure the driver was installed
* Run `python video_application.py` under dir `~/YouOnlyLookOnce-TF2.0/src/`
* The realtime video will be shown on screen

### 4.2. Image Detection

To perform detection on a list of image, we have provided a simple script to read the candidate image, perform detection, and sotre the file in local drive. \

To perform the detection: \

* Place trained model under `~/YouOnlyLookOnce-TF2.0/checkpoints`
* Use `image_application.py` under dir `~/YouOnlyLookOnce-TF2.0/src/`
* The generated image file will be stored in `~/YouOnlyLookOnce-TF2.0/tmp/single_images`

Following is the sample command for utilizing this script: \

`python image_application.py --path=/home/foo/Documents/YouOnlyLookOnce-TF2.0/test_image.jpg`







