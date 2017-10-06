#coding:utf-8
import os
import random
#import tensorflow as tf
# 首先把数据进行分类，并将其收集，方便进行索引.规定训练、验证和测试数据百分比
VALIDATION_PERCENTAGE=10
TEST_PERCENTAGE=10
#数据集位置
INPUT_LOCATION="E:"+os.sep+"1Download1"+os.sep+"tst"
def create_image_lists():
	#建立字典，按“哈希”方式”读取“所有数据（只需要组织其位置和名称即可）
	result={}
	#walk文件夹
	for cur_location,dir_names,file_names in os.walk(INPUT_LOCATION):
		train_images=[]
		validation_images=[]
		test_images=[]
		#如果当前路径下没有文件，则walk近下一个文件夹
		if not file_names:
			continue
		#否则,获得label名称，获取当前路径，并从file_names中读取图片
		else:
			label_name=os.path.basename(cur_location)
			print cur_location
			print label_name
			path=cur_location
			for filename in file_names:				
				rand=random.randint(0,100)
				if rand<VALIDATION_PERCENTAGE:
					validation_images.append(filename)
				elif rand<VALIDATION_PERCENTAGE+TEST_PERCENTAGE:
					test_images.append(filename)
				else:
					train_images.append(filename)
			#把数据存到字典里
			result[label_name]={
				'label_path':path,
				'training':train_images,
				'testing':test_images,
				'validation':validation_images
			}
	return result



def main():
	# images_lists=create_image_lists()
	# num_classes=len(image_list.keys())
	# # bottleneck的尺寸需要从“持久化模型”中获取
	# bottleneck_size=
	
	# bottleneck=[]
	# weights=tf.Variable(tf.random([]))
	# biases=tf.variable(tf.zero())
	# logits=tf.matmul(bottleneck,weights)
	# output=tf.sotfmax()
	result=create_image_lists()
	print result

if __name__=='__main__':
	main()
	
	
	
	
	
	
	
	
	
	
	
	
	