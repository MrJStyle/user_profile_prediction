#!/bin/zsh

python ./user_profile_prediction/main.py \
  --training_file_path "/Volumes/Samsung_T5/Files/Document/china_hadoop/GroupProject/project_data/data/train.csv" \
  --embedding_size 150 \
  --sentence_len 400 \
  --vocabulary_size 10000
  --min_count 1 \
  --model "TextCNN" \
  --class_num 6 \
  --label_name age \
  --learning_rate 0.001 \
  --epochs 50 \
  --batch_size 100