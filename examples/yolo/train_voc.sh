#!/usr/bin/env sh

CAFFE_HOME=../../build

SOLVER=./yolo_solver_voc.prototxt
WEIGHTS=./darknet19_448.conv.23.caffemodel

$CAFFE_HOME/tools/caffe train \
    --solver=$SOLVER --weights=$WEIGHTS --gpu=0
