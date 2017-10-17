#!/usr/bin/python
#coding:UTF-8
# make some changes
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

NUM_TRAINSET=8000#大约估计就行

REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 97000
MODEL_SAVE_PATH="/home/yzy_17/workspace/kinetics-i3d-master/data/checkpoints/rgb_imagenet"
MODEL_NAME="../data/checkpoints/rgb_imagenet/model.ckpt"

##!!myn
with open('ucf_rgb_list.json','r') as f:
    video_lists=json.load(f)
    n_classes=len(video_lists.keys())

#导入原模型，完成前馈过程
rgb_input = tf.placeholder(tf.float32,shape=(1,None,IMAGE_SIZE,IMAGE_SIZE,3))
with tf.variable_scope('RGB'):
    rgb_model = i3d.InceptionI3d(n_classes, spatial_squeeze=True, final_endpoint='Mixed_5c')
    bottleneck, _ = rgb_model(rgb_input, is_training=False, dropout_keep_prob=1.0)

variable_map = {}
for variable in tf.global_variables():
    if variable.name.split('/')[0] == 'RGB':
        variable_map[variable.name.replace(':0', '')] = variable
saver1 = tf.train.Saver(var_list=variable_map, reshape=True)

with tf.variable_scope("RGB",reuse=True):
    w_jieduan=tf.get_variable('inception_i3d/Mixed_4e/Branch_3/Conv3d_0b_1x1/batch_norm/beta')
output_conv_sg = tf.stop_gradient(w_jieduan)

#获取bottleneck(此处到时候可以加个命名空间，然后专门保存这几个变量)
bottleneck_input=tf.placeholder(tf.float32,shape=(BATCH_SIZE,None,7,7,1024),name='bottleneck_input')
groundtruth_input=tf.placeholder(tf.float32,[BATCH_SIZE,n_classes],name='groundtruth_input')


#global_step
global_step= tf.Variable(0,name='global_step',trainable=False)

#搭建全连接层(此处到时候可以在这个命名空间下，然后专门保存这几个变量)
end_point = 'UCF_Logits'
with tf.variable_scope(end_point):
    net = tf.nn.avg_pool3d(bottleneck_input, ksize=[1, 2, 7, 7, 1],strides=[1, 1, 1, 1, 1], padding=snt.VALID)
    net = tf.nn.dropout(net, 1.0)
    logits = i3d.Unit3D(output_channels=n_classes,
		    kernel_shape=[1, 1, 1],
		    activation_fn=None,
		    use_batch_norm=True,
		    use_bias=True,
		    name='Conv3d_0c_1x1')(net, is_training=True)
    logits = tf.squeeze(logits, [2, 3], name='SpatialSqueeze')#这个返回值维度是(6,7,10)
    print(logits)
    averaged_logits = tf.reduce_mean(logits, axis=1)#那么理论上此时返回的维度是(6,10)

finetune_variable_map={}
finetune_variable_list=[]
for variable in tf.global_variables():
    #收集参数，方便正则化
    if variable.name.split('/')[0] == 'UCF_Logits':
        finetune_variable_list.append(variable)
        tf.add_to_collection(tf.GraphKeys.WEIGHTS,variable)
        finetune_variable_map[variable.name.replace(':0', '')] = variable
saver2 = tf.train.Saver(var_list=finetune_variable_map, reshape=True)
#saver3 = tf.train.Saver(var_list=finetune_variable_map, reshape=True)
print("##############################################")
print(finetune_variable_map)
print("##############################################")
#softmax层
predictions = tf.nn.softmax(averaged_logits)




#定义损失函数(计算使用的是average_logits层)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=averaged_logits, labels=groundtruth_input)
regularizer=tf.contrib.layers.l2_regularizer(scale=1e-7)
regulation=tf.contrib.layers.apply_regularization(regularizer, weights_list=None)
loss=tf.reduce_mean(cross_entropy)+regulation
#学习率需要参考原文！！！！！！！	
#learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, NUM_TRAINSET / BATCH_SIZE, LEARNING_RATE_DECAY, staircase=True)

#优化
train_step = tf.train.MomentumOptimizer(LEARNING_RATE_BASE,0.9).minimize(loss, global_step=global_step,var_list=finetune_variable_list)




with tf.Session() as sess:
    #初始化变量
    tf.global_variables_initializer().run()
    #导入pre_trained模型变量
    saver1.restore(sess,MODEL_NAME)
    saver2.restore(sess,'./finetune_model/finetue.ckpt-0')

    sess.graph.finalize()
    for i in range(TRAINING_STEPS):
	x,y_ = next_batch.next_batch_bottleneck(sess,n_classes,video_lists,BATCH_SIZE,'training',rgb_input,bottleneck)#!!!!在这里读取参数
        _,loss_value, step = sess.run([train_step,loss, global_step], feed_dict={bottleneck_input: x, groundtruth_input:y_})
	if i % 1 == 0:
	    print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
        if i % 100 == 0:
            saver2.save(sess, './finetune_model/finetue.ckpt', global_step=global_step)
