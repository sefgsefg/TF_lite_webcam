#!/bin/bash

if [ $# -eq 0 ]; then
  DATA_DIR="./"
else
  DATA_DIR="$1"
fi


# Download TF Lite model with metadata.
FILE=${DATA_DIR}/efficientnet_lite0.tflite
if [ ! -f "$FILE" ]; then
  curl \
    -L 'https://tfhub.dev/tensorflow/lite-model/efficientnet/lite0/uint8/2?lite-format=tflite' \
    -o ${FILE}
fi