# VOC data to YOLO lmdb
--------
This example show how to convert VOC dataset to lmdb which is consumed by this YOLO repository.

----------
## Step By Step How To 
- ### Step 0      *Build caffe in [YOUR_LOCAL_REPOSITORY_PATH]/build*

- ### Step 1      *Download VOC2007 & 2012 dataset([here][1])* 
  ```  python
  cd [YOUR_LOCAL_REPOSITORY_PATH]/data/yolo
  wget http://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar
  wget http://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar
  wget http://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar
  wget http://pjreddie.com/media/files/VOC2012test.tar
  tar -xvf VOCtest_06-Nov-2007.tar
  tar -xvf VOCtrainval_06-Nov-2007.tar
  tar -xvf VOCtrainval_11-May-2012.tar
  tar -xvf VOC2012test.tar
  ```  
  You will get 'VOCdevkit' folder


- ### Step 2      *Create lmdb index \*.txt files* 
  ```  
  python get_list.py
  ```  
  You will get 'trainval.txt test_2007.txt test_2012.txt' files
     
- ### Step 3      *Create lmdb* 
  ```  
  ./convert.sh
  ```  
  lmdb folder has all lmdb
  
  
  [1]: https://pjreddie.com/projects/pascal-voc-dataset-mirror/
