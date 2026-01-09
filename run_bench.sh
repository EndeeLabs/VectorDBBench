#!/bin/bash

NUM_PER_BATCH=1000 DATASET_LOCAL_DIR="/home/User/vectordataset" \
vectordbbench endee \
  --token "TOKEN" \
  --region location \
  --base-url "http://127.0.0.1:8080/api/v1" \
  --index-name test_index \
  --task-label "task_detail" \
  --m 16 \
  --ef-con 128 \
  --ef-search 128 \
  --space-type cosine \
  --precision int8d \
  --version 1 \
  --case-type Performance768D1M \
  --k 30 \
  --num-concurrency "8" \
  --concurrency-duration 30 \
  --concurrency-timeout 3600 \
  --drop-old \
  --load \
  --search-concurrent \
  --search-serial
