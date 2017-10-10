#!/usr/bin/python
#coding:UTF-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_train
import mnist_inference
import os

BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH="MNIST_model/"
MODEL_NAME="model.ckpt"



#导入计算图
g=tf.train.import_meta_graph('MNIST_model/model.ckpt-6001.meta')
graph=tf.get_default_graph()

#xx = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name='xx-input')
#yy_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='yy-input')
x=graph.get_tensor_by_name('x-input:0')
y_=graph.get_tensor_by_name('y-input:0')
print(type(x))
print('\n\n\n\n\n')

print(graph.get_tensor_by_name('add:0'))
#print(tf.trainable_variables())
print(tf.global_variables())
print(graph.get_operations())
#global_step=graph.get_tensor_by_name('global_step')
train_op=graph.get_operation_by_name('train')
loss=graph.get_tensor_by_name('add:0')
global_step=tf.get_variable('yinzhiyu',dtype=tf.int32,initializer=0,trainable=False)

#print('\n\n\n')
#print(global_step)
#print(tf.get_variable('layer1/weights:0'),[784, 500])
#print('\n\n\n')

saver = tf.train.Saver()

with tf.Session() as sess:
	sess.run(global_step.initializer)
#tf.global_variables_initializer().run()	
	g.restore(sess,"MNIST_model/model.ckpt-6001")

	mnist=input_data.read_data_sets("/tmp/data",one_hot=True)

	for i in range(100000):
		xs, ys = mnist.train.next_batch(BATCH_SIZE)
		_, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={'x-input:0': xs, 'y-input:0': ys})
		if i % 1000 == 0:
			print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
			saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)
