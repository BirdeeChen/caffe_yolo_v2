## Predict VOC data by trained model
- Download the trained model from Baidu disk [https://pan.baidu.com/s/1jJ9emNW][3]
  ```
  mkdir model_voc
  cp DOWNLOAD/PATH/OF/yolo_voc_iter_120000.caffemodel model_voc
  ```
  
- Run the script:

        python detect.py
    ![image](https://github.com/quhezheng/caffe_yolo_v2/blob/master/examples/yolo/demo.jpg)
    
    detect.py use **CPU** do do predition by default, please change the script if want **GPU**  
    

## Train the VOC data
- Download the pre-trained model [darknet19_448.conv.23.caffemodel][1] from Baidu disk, copy it to this folder
  
  This model is converted directly from [darknet darknet19_448.conv.23][2]. It contain trained TOP 23 layers' weight, other layers' weight are initilized by 'xavier'
  
- Run the script:

  ```
  mkdir model_voc
  ./train_voc.sh
  ```   

 [1]: https://pan.baidu.com/s/1qZ32sJ6
 [2]: https://pjreddie.com/media/files/darknet19_448.conv.23
 [3]: https://pan.baidu.com/s/1jJ9emNW
