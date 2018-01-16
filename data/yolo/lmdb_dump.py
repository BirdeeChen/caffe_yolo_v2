sys.path.insert(0, '../../python/')
import caffe
import lmdb
import numpy as np
import cv2
from caffe.proto import caffe_pb2

def parseBoxes(float_label):
    n = len(float_label) 
    if (n % 5 > 0):
        print("Wrong size of label : {}".format(n))
        return None
    return float_label.reshape((-1, 5))



lmdb_env = lmdb.open('lmdb/trainval_lmdb')
lmdb_txn = lmdb_env.begin()
lmdb_cursor = lmdb_txn.cursor()
datum = caffe_pb2.Datum()

for key, value in lmdb_cursor:
    datum.ParseFromString(value)
    float_data = datum.float_data
    float_label = np.array(float_data).astype(float)
    label = datum.label
    #continue
    #data = caffe.io.datum_to_array(datum)

    #CxHxW to HxWxC in cv2
    #image = np.transpose(data, (1,2,0))
    image = cv2.imdecode(np.fromstring(datum.data, np.uint8), cv2.IMREAD_COLOR)
    
    boxes = parseBoxes(float_label)
    if(boxes is not None):
        width, height, _ = image.shape       
        for box in boxes:
            x = int(box[0] * width)
            y = int(box[1] * height)
            w = int(box[2] * width)
            h = int(box[3] * height)
            cv2.rectangle(image, (x,y), (x+w,y+h), (255,0,0),2)
            
    cv2.imshow('cv2', image)
    cv2.waitKey(0)
    print('{},{}'.format(key, label))
