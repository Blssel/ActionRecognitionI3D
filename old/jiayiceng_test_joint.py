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
import jiayiceng_next_batch_test 

import json

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"



IMAGE_SIZE=224


#MODEL_SAVE_PATH="/home/yzy_17/workspace/kinetics-i3d-master/data/checkpoints/rgb_imagenet"
#MODEL_NAME="../../data/checkpoints/rgb_imagenet/model.ckpt"
RGB_MODEL_NAME="/../data/checkpoints/rgb_imagenet/model.ckpt"
FLOW_MODEL_NAME="/../data/checkpoints/flow_imagenet/model.ckpt"


##读取数据list
with open('ucf_rgb_list.json','r') as f:
    video_lists=json.load(f)
    n_classes=len(video_lists.keys())


##RGB
#导入原模型，完成前馈过程
rgb_input = tf.placeholder(tf.float32,shape=(1,None,IMAGE_SIZE,IMAGE_SIZE,3))
with tf.variable_scope('RGB'):
    rgb_model = i3d.InceptionI3d(400, spatial_squeeze=True, final_endpoint='Logits')
    bottleneck, _ = rgb_model(rgb_input, is_training=False, dropout_keep_prob=1.0)

#获取bottleneck(此处到时候可以加个命名空间，然后专门保存这几个变量)
bottleneck_input=tf.placeholder(tf.float32,shape=(None,400),name='bottleneck_input')
groundtruth_input=tf.placeholder(tf.float32,[None,n_classes],name='groundtruth_input')

#搭建全连接层(此处到时候可以在这个命名空间下，然后专门保存这几个变量)
end_point = 'Fc'
with tf.variable_scope(end_point):
	weights=tf.Variable(tf.truncated_normal([400,n_classes],stddev=0.001))
	biases=tf.Variable(tf.zeros([n_classes]))
	al=tf.matmul(bottleneck_input,weights)+biases



#-------------------------------------------------------------------------------------#
#导入原模型，完成前馈过程
flow_input = tf.placeholder(tf.float32,shape=(1,None,IMAGE_SIZE,IMAGE_SIZE,2))
with tf.variable_scope('FLOW'):
        rgb_model = i3d.InceptionI3d(400, spatial_squeeze=True, final_endpoint='Logits')
        bottleneck_flow, _ = rgb_model(rgb_input, is_training=False, dropout_keep_prob=1.0)
#获取bottleneck(此处到时候可以加个命名空间，然后专门保存这几个变量)
bottleneck_input_flow=tf.placeholder(tf.float32,shape=(None,400),name='bottleneck_input_flow')
groundtruth_input_flow=tf.placeholder(tf.float32,[None,n_classes],name='groundtruth_input_flow')

#搭建全连接层(此处到时候可以在这个命名空间下，然后专门保存这几个变量)
end_point = 'Fc_flow'
with tf.variable_scope(end_point):
        weights=tf.Variable(tf.truncated_normal([400,n_classes],stddev=0.001))
        biases=tf.Variable(tf.zeros([n_classes]))
        #al=tf.nn.relu(tf.matmul(bottleneck_input,weights)+biases)
        al_flow=tf.matmul(bottleneck_input,weights)+biases

predictions1 = tf.nn.softmax(al)
predictions2 = tf.nn.softmax(al_flow)
prediction=(predictions1+predictions2)/2


#测量准确率
correct_prediction=tf.equal(tf.argmax(prediction,1),tf.argmax(groundtruth_input,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

saver_rgb=tf.train.Saver()
saver_flow=tf.train.Saver()
with tf.Session() as sess:
    #初始化变量
	#tf.global_variables_initializer().run()
    #sess.run(global_step.initializer)
	#saver2.restore(sess,'./jiayiceng_model_rgl/finetue.ckpt-2002')
    #导入pre_trained模型变量
	#saver1.restore(sess,MODEL_NAME)
    #saver2.restore(sess,'./finetune_model/finetue.ckpt-20501')
	saver_rgb.restore(sess,'./jiayiceng_model_glob/finetue2.ckpt-5402')
        saver_rgb.restore(sess,'./jiayiceng_model_flow_glob/finetune_flow.ckpt-5402')

	sess.graph.finalize()
	acc_accum=0.0
	count=0.0
	for label_index in range(len(video_lists.keys())):
		num_correct_video=0.0
		num_video=0.0
		label_name=video_lists.keys()[label_index]
		for video_index in range(len(video_lists[label_name]['validation'])):
			num_video+=1.0
			x_rgb,y_rgb,flag=jiayiceng_next_batch_test.get_bottleneck_tensor_by_indexs(sess,n_classes,video_lists,label_index,video_index,'validation', rgb_input,bottleneck)
                        x_flow,y_flow,flag_flow=jiayiceng_next_batch_test.get_bottleneck_tensor_by_indexs(sess,n_classes,video_lists,label_index,video_index,    'validation', flow_input,bottleneck_flow)
			#增加一个判断环节
			if flag==False or flag_flow==Flase:
				print('video wrong')
				continue
			else:
				count+=1.0
                        out_predictions, is_correct, acc= sess.run([prediction, correct_prediction, accuracy], feed_dict={bottleneck_input: x, groundtruth_input:y_,bottleneck_input_flow:x_flow,groundtruth_input:y_flow})
			#如果预测错误，则保存其名称,以及得分
			if is_correct==False:
				
				with open('./pre_wrong_joint.txt','a') as f:
					f.write(video_lists[label_name]['validation'][video_index])
					f.write('\n')
					out_predictions = out_predictions[0]
					sorted_indices = np.argsort(out_predictions)[::-1] 
					f.write('Top classes and probabilities\n')
					for index in sorted_indices[:20]:
						f.write(str(out_predictions[index])+' '+video_lists.keys()[index]+'\n')
					f.write('\n\n')

			else:
				num_correct_video+=1.0
			acc_accum=acc_accum+acc
		with open('./pre_wrong_pre_flow_ucf.txt','a') as ff:
			#ff.write('###########################\n')
			#ff.write('The class '+ label_name +' has ' +str(len(video_lists[label_name]['validation']))+' videos , ' +str(int(num_correct_video)) +' right, and the accuracy is :\n')
			ff.write('{:<22}'.format(label_name))
			ff.write(str(num_correct_video/num_video)+'\n')
			#ff.write('##########################\n')	
	acc=acc_accum/count
	print(acc)
