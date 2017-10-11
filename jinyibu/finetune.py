#!/usr/bin/python
#coding:UTF-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
	   
import i3d
import next_batch 

BATCH_SIZ = 6
LEARNING_RATE_BASE = 0.9
LEARNING_RATE_DECAY = 0.99

NUM_TRAINSET

REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH="i3d_models/"
MODEL_NAME="ucf_finetune_model.ckpt"

##!!myn
video_lists=
n_classes=len(video_lists.keys())

#导入原模型，完成前馈过程
rgb_input = tf.placeholder(tf.float32,shape=None)
rgb_model = i3d.InceptionI3d(n_classes, spatial_squeeze=True, final_endpoint='Mixed_5c')
bottleneck, _ = rgb_model(rgb_input, is_training=False, dropout_keep_prob=1.0)


#获取bottleneck
bottleneck_input=tf.placeholder(tf.float32,shape=None,name='BottleneckInputPlaceholder')
groundtruth_input=tf.placeholder(tf.float32,[None,n_classes],name='GroundTruthInput')


#搭建全连接层
end_point = 'UCF_Logits'
with tf.variable_scope(end_point):
	net = tf.nn.avg_pool3d(bottleneck_input, ksize=[1, 2, 7, 7, 1],strides=[1, 1, 1, 1, 1], padding=snt.VALID)
	net = tf.nn.dropout(net, dropout_keep_prob)
	logits = Unit3D(output_channels=UCF_NUM_CLASS,
					kernel_shape=[1, 1, 1],
					activation_fn=None,
					use_batch_norm=True,
					use_bias=True,
					name='Conv3d_0c_1x1')(net, is_training=True)
	logits = tf.squeeze(logits, [2, 3], name='SpatialSqueeze')##这个squeeze究竟是什么？？？？？？
	averaged_logits = tf.reduce_mean(logits, axis=1)

#softmax层
predictions = tf.nn.softmax(averaged_logits)

#global_step
global_step = tf.Variable(0,name='global_step',trainable=False)


#定义损失函数(计算使用的是average_logits层)
loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=averaged_logits, labels=tf.argmax(groundtruth_input, 1))

#学习率需要参考原文！！！！！！！	
learning_rate = tf.train.exponential_decay(
	LEARNING_RATE_BASE,
	global_step,
	NUM_TRAINSET / BATCH_SIZE, LEARNING_RATE_DECAY,
	staircase=True)

#优化
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

#Saver类	
saver=tf.train.Saver(reshape=True)



with tf.Session() as sess:
	#初始化变量
	tf.global_variables_initializer().run()
	#导入pre_trained模型变量
	saver.restore(sess,MODEL_NAME)

	for i in range(TRAINING_STEPS):
		x,y_ = next_batch.next_batch_bottleneck(sess,n_classes,video_lists,BATCH_SIZE,'training',rgb_input,bottleneck)#!!!!在这里读取参数
		loss_value, step = sess.run([loss, global_step], feed_dict={bottleneck_input: x, groundtruth_input:y_})
		if i % 1000 == 0:
			print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
#saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)
