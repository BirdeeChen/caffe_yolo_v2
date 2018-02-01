import numpy as np
import cv2
import os, sys
import Image
import ImageDraw
import ImageFont
import matplotlib.image as mpimg
sys.path.insert(0, '../python/')
import caffe
#caffe.set_device(0)
caffe.set_mode_cpu()

mean = np.require([104, 117, 123], dtype=np.float32)[:, np.newaxis, np.newaxis]

def get_names_from_file(filename):
    result = []
    fd = file(filename, 'r')
    for line in fd.readlines():
        result.append(line.replace('\n', ''))
    return result


def get_color_from_file(filename):
    colors = []
    fd = file(filename, 'r')
    for line in fd.readlines():
        words = line.split(r',')
        color = (int(words[0]), int(words[1]), int(words[2]))
        colors.append(color)
    return colors


def draw_image(pic_name, boxes, namelist_file):
    name_list = get_names_from_file(namelist_file)
    color_list = get_color_from_file('ink.color')
    im = Image.open(pic_name)
    draw = ImageDraw.Draw(im)
    lena = mpimg.imread(pic_name)
    height, width = lena.shape[:2]
    for box in boxes:
        x = box[3]
        y = box[4]
        w = box[5]
        h = box[6]
        left = (x - w / 2) * width
        right = (x + w / 2) * width
        top = (y - h / 2) * height
        bot = (y + h / 2) * height
        if left < 0:
            left = 0
        if right > width - 1:
            right = width - 1
        if top < 0:
            top = 0
        if bot > height - 1:
            bot = height - 1
        category_id = int(box[1])
        category = name_list[category_id]
        color = color_list[category_id % color_list.__len__()]
        draw.line((left, top, right, top), fill=color, width=5)
        draw.line((right, top, right, bot), fill=color, width=5)
        draw.line((left, top, left, bot), fill=color, width=5)
        draw.line((left, bot, right, bot), fill=color, width=5)
        font_size = 20
        my_font = ImageFont.truetype("/usr/share/fonts/truetype/ubuntu-font-family/Ubuntu-M.ttf", size=font_size)
        draw.text([left + 5, top], category, font=my_font, fill=color)
    im.show()

def det(model, im_path, show=0):
  '''forward processing'''
  image = caffe.io.load_image(im_path)
  transformer = caffe.io.Transformer({'data': (1, 3, 416, 416)})
  transformer.set_transpose('data', (2, 0, 1))  # move image channels to outermost dimension
  transformed_image = transformer.preprocess('data', image)
  model.blobs['data'].reshape(1, 3, 416, 416)
  model.blobs['data'].data[...] = transformed_image
  out_blobs = model.forward()
  reg_out = model.blobs["Output"].data[0]
  print reg_out
  #region1 = out_blobs["region1"]
  #print region1
  return reg_out

if __name__=="__main__":
  net_proto = "./yolo_deploy.prototxt"
  model_path = "./yolo.caffemodel"
  im_path = "data/dog.jpg"
  if sys.argv.__len__() >= 3:
    net_proto = sys.argv[1]
    model_path = sys.argv[2]
    im_path = sys.argv[3]
  model = caffe.Net(net_proto, caffe.TEST, weights=model_path)

  boxes = det(model, im_path)
  boxes = boxes.reshape([-1, boxes.shape[-1]])
  draw_image(im_path, boxes, namelist_file='coco.names')

