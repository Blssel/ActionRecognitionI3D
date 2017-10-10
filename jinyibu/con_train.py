#!/usr/bin/python
#coding:UTF-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
	   
import i3d

BATCH_SIZE = 6
LEARNING_RATE_BASE = 0.9
LEARNING_RATE_DECAY = 0.99

NUM_TRAINSET

REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99:q!:
MODEL_SAVE_PATH="UCF_finetune_models/"
MODEL_NAME="ucf_finetune_model.ckpt"

#导入计算图
g=tf.train.import_meta_graph('')
graph=tf.get_default_graph()

#获取bottleneck
bottleneck=global_step=tf.get_variable('Mixed_5c')#!!!!!!这里是bottleneck
bottleneck=p[laceholder(shape)


#搭建全连接层
end_point = 'UCF_Logits'
with tf.variable_scope(end_point):
	net = tf.nn.avg_pool3d(net, ksize=[1, 2, 7, 7, 1],strides=[1, 1, 1, 1, 1], padding=snt.VALID)
	net = tf.nn.dropout(net, dropout_keep_prob)
	logits = Unit3D(output_channels=UCF_NUM_CLASS,
					kernel_shape=[1, 1, 1],
					activation_fn=None,
					use_batch_norm=True,
					use_bias=True,
					name='Conv3d_0c_1x1')(bottleneck, is_training=True)
	logits = tf.squeeze(logits, [2, 3], name='SpatialSqueeze')
	averaged_logits = tf.reduce_mean(logits, axis=1)

#softmax层
predictions = tf.nn.softmax(averaged_logits)
global_step = tf.Variable(0,name='global_step',trainable=False)


#如果不导入计算图，则100%确定只优化全脸阶层
#定义损失函数(计算使用的是average_logits层)
loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=averaged_logits, labels=tf.argmax(y_, 1))
learning_rate = tf.train.exponential_decay(
	LEARNING_RATE_BASE,
	global_step,
	NUM_TRAINSET / BATCH_SIZE, LEARNING_RATE_DECAY,
	staircase=True)
#优化
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

saver = tf.train.Saver()



with tf.Session() as sess:
	#初始化变量
	tf.global_variables_initializer().run()
	for i in range(TRAINING_STEPS):
		xs, ys = next_batch(BATCH_SIZE,,,)#!!!!在这里读取参数
		loss_value, step = sess.run([loss, global_step], feed_dict={x: xs, y_: ys})
		if i % 1000 == 0:
			print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
			saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)
