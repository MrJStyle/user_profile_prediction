#!/bin/zsh

MODEL_PATH=$1;
MODEL_NAME=$2;

docker run -it --rm \
-p 8501:8501 \
-v ${MODEL_PATH}/${MODEL_NAME}:/models/${MODEL_NAME} \
-e MODEL_NAME=${MODEL_NAME} \
tensorflow/serving;