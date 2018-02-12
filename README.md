# YOLO: Real-Time Object Detection (Caffe)

This is the caffe version YOLO V2 ported directly from [darknet YOLO][1]

## Detection Using A Pre-Trained VOC Model

- ### Step 0 *BUILD this reporsitory caffe*
  ```
  git clone https://github.com/quhezheng/caffe_yolo_v2
  cd caffe_yolo_v2
  mkdir build
  cd build
  cmake ..
  make  
  ```

- ### Step 1 *Download the trained model from Baidu disk  [https://pan.baidu.com/s/1jJ9emNW][3]*
  ```
  cd ..  #back to project root folder
  cd examples/yolo
  mkdir model_voc
  cp DOWNLOAD/PATH/OF/yolo_voc_iter_120000.caffemodel model_voc
  python detect.py
  ```
  ![image](https://github.com/quhezheng/caffe_yolo_v2/blob/master/examples/yolo/demo.jpg)
    
    detect.py use **CPU** do do predition by default, please change the script if want **GPU**  

## Train the VOC data

- ### Step 0      *Build caffe in [YOUR_LOCAL_REPOSITORY_PATH]/build*

- ### Step 1      *Download VOC2007 & 2012 dataset([here][4])* 
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
  There it will be 'VOCdevkit' folder


- ### Step 2      *Create lmdb index \*.txt files* 
  ```  
  python get_list.py
  ```  
  There it will be 'trainval.txt test_2007.txt test_2012.txt' files
     
- ### Step 3      *Create lmdb* 
  ```  
  ./convert.sh
  ```  
   There it will be lmdb folder has all lmdb

- ### Step 4 *Download the pre-trained model [darknet19_448.conv.23.caffemodel][2] from Baidu disk*

  This model is converted directly from [darknet darknet19_448.conv.23][5]. It contain trained TOP 23 layers' weight, other layers' weight are initilized by 'xavier'
  
  ```  
  cd ../../examples/yolo
  cp DOWNLOAD/PATH/OF/darknet19_448.conv.23.caffemodel ./
  mkdir model_voc
  ./train_voc.sh
  ```  
  

 [1]: https://pjreddie.com/darknet/yolo
 [2]: https://pan.baidu.com/s/1qZ32sJ6
 [3]: https://pan.baidu.com/s/1jJ9emNW
 [4]: https://pjreddie.com/projects/pascal-voc-dataset-mirror/
 [5]: https://pjreddie.com/media/files/darknet19_448.conv.23
