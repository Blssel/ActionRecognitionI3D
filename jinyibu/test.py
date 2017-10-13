#coding:UTF-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import sonnet as snt

import i3d
import next_batch

import json

import random
BATCH_SIZE = 6
IMAGE_SIZE=224

LEARNING_RATE_BASE = 0.9
LEARNING_RATE_DECAY = 0.99

NUM_TRAINSET=40#临时写的

REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH="~/workspace/kinetics-i3d-master/data/checkpoints/rgb_imagenet"
MODEL_NAME="data/checkpoints/rgb_imagenet/model.ckpt"


##!!myn
with open('result1.json','r') as f:
    video_lists=json.load(f)
    n_classes=len(video_lists.keys())

#导入原模型，完成前馈过程
rgb_input = tf.placeholder(tf.float32,shape=(1,64,IMAGE_SIZE,IMAGE_SIZE,3))

with tf.variable_scope('RGB'):
    rgb_model = i3d.InceptionI3d(n_classes, spatial_squeeze=True, final_endpoint='Mixed_5c')
    bottleneck, _ = rgb_model(rgb_input, is_training=False, dropout_keep_prob=1.0)

variable_map = {}
for variable in tf.global_variables():
    if variable.name.split('/')[0] == 'RGB':
        variable_map[variable.name.replace(':0', '')] = variable
saver = tf.train.Saver(var_list=variable_map, reshape=True)


with tf.Session() as sess:
    #初始化变量
    tf.global_variables_initializer().run()
    #导入pre_trained模型变量
    #saver.restore(sess,MODEL_NAME)
    x,y_ = next_batch.next_batch_bottleneck(sess,n_classes,video_lists,BATCH_SIZE,'training',rgb_input,bottleneck)#!!!!在这里读取参数
    print(type(x))
    print(type(y_))
    print(x.shape)
    print(y_.shape)
