#!/usr/bin/python2
#coding:UTF-8
# make some changes
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import sonnet as snt
       
import i3d
import jiayiceng_next_batch 

import json

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"



BATCH_SIZE = 500#尝试6,12,24...(论文是6)
IMAGE_SIZE=224

NUM_TRAINSET=8000#大约估计就行

TRAINING_STEPS = 80000

LEARNING_RATE_BASE = 0.001#学习率 0.1*0.99^(500000/1000)(论文学习率是1e-1 1e-2 1e-3)
LEARNING_DECAY_RATE=0.99
DECAY_STEP=100


#MODEL_SAVE_PATH="/home/yzy_17/workspace/kinetics-i3d-master/data/checkpoints/rgb_imagenet"
#MODEL_NAME="../../data/checkpoints/rgb_imagenet/model.ckpt"
MODEL_NAME="../data/checkpoints/flow_imagenet/model.ckpt"



##读取数据list
with open('ucf_rgb_list.json','r') as f:
    video_lists=json.load(f)
    n_classes=len(video_lists.keys())



#导入原模型，完成前馈过程
flow_input = tf.placeholder(tf.float32,shape=(1,None,IMAGE_SIZE,IMAGE_SIZE,2))
with tf.variable_scope('FLOW'):
    rgb_model = i3d.InceptionI3d(400, spatial_squeeze=True, final_endpoint='Logits')
    bottleneck_flow, _ = rgb_model(rgb_input, is_training=False, dropout_keep_prob=1.0)


variable_map = {}
for variable in tf.global_variables():
    if variable.name.split('/')[0] == 'FLOW':
        variable_map[variable.name.replace(':0', '')] = variable
saver_flow_1 = tf.train.Saver(var_list=variable_map, reshape=True)


#获取bottleneck(此处到时候可以加个命名空间，然后专门保存这几个变量)
bottleneck_input_flow=tf.placeholder(tf.float32,shape=(None,400),name='bottleneck_input_flow')
groundtruth_input_flow=tf.placeholder(tf.float32,[None,n_classes],name='groundtruth_input_flow')


#global_step
global_step= tf.Variable(0,name='global_step_flow',trainable=False)


#搭建全连接层(此处到时候可以在这个命名空间下，然后专门保存这几个变量)
end_point = 'Fc_flow'
with tf.variable_scope(end_point):
	weights=tf.Variable(tf.truncated_normal([400,n_classes],stddev=0.001))
	biases=tf.Variable(tf.zeros([n_classes]))
	#al=tf.nn.relu(tf.matmul(bottleneck_input,weights)+biases)
	al_flow=tf.matmul(bottleneck_input,weights)+biases


finetune_variable_map={}
finetune_variable_list=[]#需要更新的参数，最后测试时使用这部分参数（即除去momentum等以外的参数）
for variable in tf.global_variables():
    if variable.name.split('/')[0] == 'Fc_flow':
        finetune_variable_list.append(variable)
        tf.add_to_collection(tf.GraphKeys.WEIGHTS,variable)
        finetune_variable_map[variable.name.replace(':0', '')] = variable
saver_flow_2 = tf.train.Saver(var_list=finetune_variable_map,reshape=True)#saver2负责保存或加载仅全连接层的计算图和参数，并不涉及优化部分
print('#################')
print(finetune_variable_list)
print('################')


predictions = tf.nn.softmax(al_flow)


#定义损失函数(计算使用的是average_logits层)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=al_flow, labels=groundtruth_input)
regularizer=tf.contrib.layers.l2_regularizer(scale=1e-7)
regulation=tf.contrib.layers.apply_regularization(regularizer, weights_list=None)
loss=tf.reduce_mean(cross_entropy)+regulation
#loss=tf.reduce_mean(cross_entropy)
	
learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step-0, DECAY_STEP, LEARNING_DECAY_RATE, staircase=True)

#优化(只更新新建部分的参数)
#train_step = tf.train.MomentumOptimizer(learning_rate,0.9).minimize(loss, global_step=global_step,var_list=finetune_variable_list)
train_step = tf.train.MomentumOptimizer(learning_rate,0.9).minimize(loss, global_step=global_step,var_list=tf.global_variables())

#收集变量，唯一的作用就是方便继续训练使用（但如果修改了learning_rate的话,继续训练时就不应该沿用上一次的momentum）
finetune_variable_map_including_optimizer={}#此处字典保存了包括全连接层在内的所有变量（即包括了momentum）
for variable in tf.global_variables():
    if variable.name.split('/')[0] == 'Fc':
        finetune_variable_map_including_optimizer[variable.name.replace(':0', '')] = variable
saver_flow_3 = tf.train.Saver(var_list=finetune_variable_map_including_optimizer,reshape=True)
print('#################')
for key in finetune_variable_map_including_optimizer.keys():
    print(finetune_variable_map_including_optimizer[key])
print('################')
saver_flow_4=tf.train.Saver(var_list=tf.global_variables())
with tf.Session() as sess:
    #初始化变量
    tf.global_variables_initializer().run()
    #sess.run(global_step.initializer)
	#saver2.restore(sess,'./finetune_model/finetue.ckpt-0')
    #导入pre_trained模型变量
    saver1.restore(sess,MODEL_NAME)
	#saver2.restore(sess,'./jiayiceng_model_rgl/finetue.ckpt-2002')
    #saver4.restore(sess,'./jiayiceng_model_glob/finetue2.ckpt-8603')

    sess.graph.finalize()
    for i in range(TRAINING_STEPS):
		x,y_ = jiayiceng_next_batch.next_batch_bottleneck(sess,n_classes,video_lists,BATCH_SIZE,'training',flow_input,bottleneck_flow)#!!!!在这里读取参数
		#print(x[0])
		_,loss_value, step ,p= sess.run([train_step,loss, global_step,predictions], feed_dict={bottleneck_input_flow: x, groundtruth_input_flow:y_})
		#print(p[0])
		#print(y_[0])
		if i % 30 == 0:
			print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
		if i % 250 == 0:
			saver_flow_4.save(sess, './jiayiceng_model_flow_glob/finetune_flow.ckpt', global_step=global_step)

