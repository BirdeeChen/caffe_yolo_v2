import numpy as np
import cv2
import os, sys
sys.path.insert(0, '../python/')
import caffe
#caffe.set_device(0)
caffe.set_mode_cpu()

mean = np.require([104, 117, 123], dtype=np.float32)[:, np.newaxis, np.newaxis]

def det(model, im_path, show=0):
  '''forward processing'''
  im = cv2.imread(im_path)
  im = cv2.resize(im, (608, 608))
  im = np.require(im.transpose((2, 0, 1)), dtype=np.float32)
  im -= mean
  model.blobs['data'].data[...] = im
  out_blobs = model.forward()
  reg_out = out_blobs["Output"]
  print reg_out

if __name__=="__main__":
  #net_proto = "./yolo.prototxt"
  #model_path = "./yolo.caffemodel"
  net_proto = sys.argv[1]
  model_path = sys.argv[2]
  model = caffe.Net(net_proto, caffe.TEST, weights=model_path)

  im_path = sys.argv[3]
  det(model, im_path)