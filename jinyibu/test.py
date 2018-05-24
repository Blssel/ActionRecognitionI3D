#!/usr/bin/python
#coding:UTF-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import sonnet as snt
	   
import i3d
import next_video 

import json

BATCH_SIZE = 6
IMAGE_SIZE=224

LEARNING_RATE_BASE = 1e-2      #1e-1   1e-2 1e-3        0.36  0.135   0.0498
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
bottleneck_input=tf.placeholder(tf.float32,shape=(1,None,7,7,1024),name='bottleneck_input')
groundtruth_input=tf.placeholder(tf.float32,[1,n_classes],name='groundtruth_input')


#global_step
global_step= tf.Variable(97000,name='global_step',trainable=False)

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
    averaged_logits = tf.reduce_mean(logits, axis=1)#那么理论上此时返回的维度是(6,10)i

#获取finetune层的参数
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
#print("##############################################")
#print(finetune_variable_map)
#print("##############################################")

#softmax层
predictions = tf.nn.softmax(averaged_logits)

#测量准确率
correct_prediction=tf.equal(tf.argmax(predictions,1),tf.argmax(groundtruth_input,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))


with tf.Session() as sess:
    #初始化变量
    tf.global_variables_initializer().run()

    #导入finetune部分的模型 
    #saver2.restore(sess,'./finetune_model/finetue.ckpt-41001')

    #导入pre_trained模型变量
    saver1.restore(sess,MODEL_NAME)


#顺序挑选每一个视频，并计算bottleneck
    for label_index in range(len(video_lists.keys())):
        #生成这个类对应的groundtruth编码（维度就是类别数）
        ground_truth = np.zeros([1,n_classes], dtype=np.float32)#数据类型是float32型
        ground_truth[0][label_index] = 1.0#让这个类的索引对应位的值取1

        label_name=video_lists.keys()[label_index]
        for video_index in range(len(video_lists[label_name]['testing'])):
            #计算bottleneck的值!!!!!
            bottleneck,flag = next_video.get_or_create_bottleneck(sess, video_lists, label_name, video_index, 'testing', rgb_input,bottleneck)
            if flag==False:
                #print('format is not suitable,pass!----%s'%label_name)
                continue
            
            x = bottleneck
            y_ = ground_truth
            print(x.shape)
            print(y_.shape)
            #accuracy_score = sess.run(accuracy, feed_dict={bottleneck_input: x, groundtruth_input:y_})
    #print(accuracy_score)
