#!/usr/bin/python
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

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"



BATCH_SIZE = 12#尝试6,12,24...(论文是6)
IMAGE_SIZE=224

NUM_TRAINSET=8000#大约估计就行

TRAINING_STEPS = 500000

LEARNING_RATE_BASE = 0.1#学习率 0.1*0.99^(500000/1000)(论文学习率是1e-1 1e-2 1e-3)
LEARNING_DECAY_RATE=0.99
DECAY_STEP=1000


#MODEL_SAVE_PATH="/home/yzy_17/workspace/kinetics-i3d-master/data/checkpoints/rgb_imagenet"
MODEL_NAME="../data/checkpoints/rgb_imagenet/model.ckpt"



##读取数据list
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
finetune_variable_list=[]#需要更新的参数，最后测试时使用这部分参数（即除去momentum等以外的参数）
for variable in tf.global_variables():
    if variable.name.split('/')[0] == 'UCF_Logits':
        finetune_variable_list.append(variable)
        tf.add_to_collection(tf.GraphKeys.WEIGHTS,variable)
        finetune_variable_map[variable.name.replace(':0', '')] = variable
saver2 = tf.train.Saver(var_list=finetune_variable_map,reshape=True)#saver2负责保存或加载仅全连接层的计算图和参数，并不涉及优化部分
print('#################')
print(finetune_variable_list)
print('################')


predictions = tf.nn.softmax(averaged_logits)


#定义损失函数(计算使用的是average_logits层)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=averaged_logits, labels=groundtruth_input)
regularizer=tf.contrib.layers.l2_regularizer(scale=1e-7)
regulation=tf.contrib.layers.apply_regularization(regularizer, weights_list=None)
loss=tf.reduce_mean(cross_entropy)+regulation
#loss=tf.reduce_mean(cross_entropy)
	
learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step-0, DECAY_STEP, LEARNING_DECAY_RATE, staircase=True)

#优化(只更新新建部分的参数)
train_step = tf.train.MomentumOptimizer(learning_rate,0.9).minimize(loss, global_step=global_step,var_list=finetune_variable_list)

#收集变量，唯一的作用就是方便继续训练使用（但如果修改了learning_rate的话,继续训练时就不应该沿用上一次的momentum）
finetune_variable_map_including_optimizer={}#此处字典保存了包括全连接层在内的所有变量（即包括了momentum）
for variable in tf.global_variables():
    if variable.name.split('/')[0] == 'UCF_Logits':
        finetune_variable_map_including_optimizer[variable.name.replace(':0', '')] = variable
saver3 = tf.train.Saver(var_list=finetune_variable_map_including_optimizer,reshape=True)
print('#################')
for key in finetune_variable_map_including_optimizer.keys():
    print(finetune_variable_map_including_optimizer[key])
print('################')

with tf.Session() as sess:
    #初始化变量
    tf.global_variables_initializer().run()
    #sess.run(global_step.initializer)
    #saver2.restore(sess,'./finetune_model/finetue.ckpt-0')
    #导入pre_trained模型变量
    saver1.restore(sess,MODEL_NAME)
    #saver2.restore(sess,'./finetune_model/finetue.ckpt-20501')

    sess.graph.finalize()
    for i in range(TRAINING_STEPS):
	x,y_ = next_batch.next_batch_bottleneck(sess,n_classes,video_lists,BATCH_SIZE,'training',rgb_input,bottleneck)#!!!!在这里读取参数
        _,loss_value, step = sess.run([train_step,loss, global_step], feed_dict={bottleneck_input: x, groundtruth_input:y_})
	if i % 5 == 0:
	    print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
        if i % 1000 == 0:
            saver2.save(sess, './finetune_model/finetue.ckpt', global_step=global_step)
