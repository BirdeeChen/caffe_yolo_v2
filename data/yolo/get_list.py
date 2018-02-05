#!/usr/bin/env python
import os

def Append_jpg_xml_lists(jpeg_fold, xml_fold, file_path, jpeg_list, xml_list, ignoreXML=False):
  with open(file_path, 'r') as fp:
    for line in fp:
      line = line.strip()
      jpeg = os.path.join(jpeg_fold, "{}.jpg".format(line))
      xml = os.path.join(xml_fold, "{}.xml".format(line))
    
      if not os.path.exists(jpeg):
        print jpeg, "not exist"
        continue      

      if not os.path.exists(xml):
        if not ignoreXML:
          print xml, "not exist"
          continue
        xml = ""

      jpeg_list.append(jpeg)
      xml_list.append(xml)

trainval_jpeg_list = []
trainval_xml_list = []
test07_jpeg_list = []
test07_xml_list = []
test12_jpeg_list = []
test12_xml_list = []

for name in ["VOC2007", "VOC2012"]:
  voc_dir = os.path.join("VOCdevkit", name)
  txt_fold = os.path.join(voc_dir, "ImageSets/Main")
  jpeg_fold = os.path.join(voc_dir, "JPEGImages")
  xml_fold = os.path.join(voc_dir, "Annotations")
  for t in ["train.txt", "val.txt"]:
    file_path = os.path.join(txt_fold, t)
    Append_jpg_xml_lists(jpeg_fold, xml_fold, file_path, trainval_jpeg_list, trainval_xml_list)

    '''
    with open(file_path, 'r') as fp:
      for line in fp:
        line = line.strip()
        trainval_jpeg_list.append(os.path.join(jpeg_fold, "{}.jpg".format(line)))
        trainval_xml_list.append(os.path.join(xml_fold, "{}.xml".format(line)))
        if not os.path.exists(trainval_jpeg_list[-1]):
          print trainval_jpeg_list[-1], "not exist"
        if not os.path.exists(trainval_xml_list[-1]):
          print trainval_xml_list[-1], "not exist"
    '''
  file_path = os.path.join(txt_fold, "test.txt")
  if not os.path.exists(file_path):
    print "test.txt for ", name, " not found, ignored and continue..."
    continue
  if name == "VOC2007":
    Append_jpg_xml_lists(jpeg_fold, xml_fold, file_path, test07_jpeg_list, test07_xml_list)
  elif name == "VOC2012":
    Append_jpg_xml_lists(jpeg_fold, xml_fold, file_path, test12_jpeg_list, test12_xml_list, True)
  '''
  if name == "VOC2007":    
    with open(file_path, 'r') as fp:
      for line in fp:
        line = line.strip()
        test07_jpeg_list.append(os.path.join(jpeg_fold, "{}.jpg".format(line)))
        test07_xml_list.append(os.path.join(xml_fold, "{}.xml".format(line)))
        if not os.path.exists(test07_jpeg_list[-1]):
          print test07_jpeg_list[-1], "not exist"
        if not os.path.exists(test07_xml_list[-1]):
          print test07_xml_list[-1], "not exist"
  elif name == "VOC2012":    
    with open(file_path, 'r') as fp:
      for line in fp:
        line = line.strip()
        test12_jpeg_list.append(os.path.join(jpeg_fold, "{}.jpg".format(line)))
        test12_xml_list.append(os.path.join(xml_fold, "{}.xml".format(line)))
        if not os.path.exists(test12_jpeg_list[-1]):
          print test12_jpeg_list[-1], "not exist"
        if not os.path.exists(test12_xml_list[-1]):
          print test12_xml_list[-1], "not exist"
  '''

with open("trainval.txt", "w") as wr:
  for i in range(len(trainval_jpeg_list)):
    wr.write("{} {}\n".format(trainval_jpeg_list[i], trainval_xml_list[i]))
  print "trainval.txt created successfully !"

with open("test_2007.txt", "w") as wr:
  for i in range(len(test07_jpeg_list)):
    wr.write("{} {}\n".format(test07_jpeg_list[i], test07_xml_list[i]))
  if len(test07_jpeg_list) > 0:
    print "test_2007.txt created successfully !"

with open("test_2012.txt", "w") as wr:
  for i in range(len(test12_jpeg_list)):
    wr.write("{} {}\n".format(test12_jpeg_list[i], test12_xml_list[i]))
  if len(test12_jpeg_list) > 0:
    print "test_2012.txt created successfully !"


