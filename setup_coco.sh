# Setup, this may take over an hour

# mkdir
mkdir data/coco
mkdir data/coco/annotations
mkdir data/coco/images

# Data Download and Unzip
cd data/coco/annotations
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

unzip annotations_trainval2017.zip

cd ../images
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip

unzip train2017.zip
unzip val2017.zip

# Move coco.names to data
mv ../../../tool/coco.names ../../../data/

# Generate Label files
cd ../../../tool
python preprocess_coco.py

# Start training
# cd ../src/
# python main.py
