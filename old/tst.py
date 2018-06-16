from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
import sonnet as snt
import i3d
import next_batch
import json
BATCH_SIZE = 6
IMAGE_SIZE=224
LEARNING_RATE_BASE = 1e-1
LEARNING_RATE_DECAY = 0.99
NUM_TRAINSET=8000
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 97000
MODEL_SAVE_PATH="/home/yzy_17/workspace/kinetics-i3d-master/data/checkpoints/rgb_imagenet"
MODEL_NAME="../data/checkpoints/rgb_imagenet/model.ckpt"
##!!myn
with open('ucf_rgb_list.json','r') as f:
    video_lists=json.load(f)
    n_classes=len(video_lists.keys())



g=tf.train.import_meta_graph('./finetune_model/finetue.ckpt-1001.meta')
graph=tf.get_default_graph()

print(tf.global_variables())

