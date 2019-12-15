# Automatically setup for VOC dataset
pip install -r requirements.txt

# Download PASCAL 2007/2012
cd data
wget https://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar
wget https://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar
wget https://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar

# Unzip
tar xf VOCtrainval_11-May-2012.tar
tar xf VOCtrainval_06-Nov-2007.tar
tar xf VOCtest_06-Nov-2007.tar

# Generate PASCAL lists
wget https://pjreddie.com/media/files/voc_label.py
python voc_label.py

# Generate PASCAL lists for YOLO
cd ../tool
python preprocess_pascal_voc.py

# Finish
cd ../src

# Start Training
python main.py
