## Tool convert [Darknet][1] model to caffe

 1. This is the tool to convert [Darknet][1] YOLO V2 *.cfg & *.weight file to caffe *.prototxt & *.caffemodel file
 2. The script to predict picture by caffe in this repository is also provided.


----------
## Demo Ouput 
![image](https://github.com/quhezheng/caffe_yolo_v2/blob/master/darknet/demo.jpg)
----------
## Step By Step How To 

- ## Step 1     *Down load darknet \*.weight file [here(256M)][2]* 

         wget https://pjreddie.com/media/files/yolo.weights
 
- ## Step 2  *Convert to yolo.caffemodel*
 
        python darknet2caffe.py yolo.cfg yolo.weights yolo.prototxt yolo.caffemodel
   
     **yolo.prototxt** & **yolo.caffemodel** are created


- ## Step 3  *Add **DetectionOutput** or **Region** layer to do prediction*

  Open the created yolo.prototxt and append the following layer code to the end, save it as ***yolo_deploy.prototxt***
  
```   
 layer {
    bottom: "layer31-conv"
    top: "Output"
    name: "Output"
    type: "DetectionOutput"
    detection_output_param {
        classes: 80
        confidence_threshold: 0.2
        nms_threshold: 0.45
        biases: 0.57273
        biases: 0.677385
        biases: 1.87446
        biases: 2.06253
        biases: 3.33843
        biases: 5.47434
        biases: 7.88282
        biases: 3.52778
        biases: 9.77052
        biases: 9.16828
    }
}

layer {
  name: "region1"
  type: "Region"
  bottom: "layer31-conv"
  top: "region1"
  region_param {
    classes: 80
    coords: 4
    boxes_of_each_grid: 5
    softmax: true
  }
}
``` 
 - [x] yolo_deploy.prototxt is already avaiable in the folder
 - [x] Choose either "DetectionOutput" or "Region" is OK. "DetectionOutput" is recommended

 - ## Step 4  Execute the prediction (Forward only)

      - Option 1  (When "DetectionOutput" layer added in step 3)

            python detect.py
    or
    
            python detect.py yolo_deploy.prototxt yolo.caffemodel data/person.jpg 
    to choose the model and picture
          
      - Option 2  (When "Region" layer added in step 3)
      
             python detect_region_layer.py
   If want to choose model and picture, pls change the code in detect_region_layer.py self
   
   
   
   
  [1]: https://pjreddie.com/darknet/yolo/
  [2]: https://pjreddie.com/media/files/yolo.weights
