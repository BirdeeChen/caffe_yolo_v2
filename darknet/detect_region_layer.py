import os, sys
sys.path.insert(0, '../python/')
import caffe
import numpy as np

import region_tool as tool

pic_name = "data/dog.jpg"#'dog.jpg'#'person.jpg'
# caffe.set_device(0)
# caffe.set_mode_gpu()
caffe.set_mode_cpu()
image = caffe.io.load_image(pic_name)
transformer = caffe.io.Transformer({'data': (1, 3, 416, 416)})
transformer.set_transpose('data', (2, 0, 1))  # move image channels to outermost dimension
transformed_image = transformer.preprocess('data', image)
print transformed_image.shape

#model_def = 'yolo_revised.prototxt'
#model_weights = 'yolo.caffemodel'
model_def = "yolo_deploy.prototxt"
model_weights = 'yolo.caffemodel'

net = caffe.Net(model_def, model_weights, caffe.TEST)
net.blobs['data'].reshape(1, 3, 416, 416)
net.blobs['data'].data[...] = transformed_image
output = net.forward()
feat = net.blobs['region1'].data[0]
print feat.shape

boxes_of_each_grid = 5
classes = 80
thread = 5.945
biases = np.array([0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828])
boxes = tool.get_region_boxes(feat, boxes_of_each_grid, classes, thread, biases)

for box in boxes:
    print box

tool.draw_image(pic_name, boxes=boxes, namelist_file='coco.names')
