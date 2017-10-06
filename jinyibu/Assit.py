#coding:utf-8
import os
import random

# 首先把数据进行分类，并将其收集，方便进行索引.规定训练、验证和测试数据百分比
VALIDATION_PERCENTAGE=10
TEST_PERCENTAGE=10
#数据集和bottleneck位置
INPUT_LOCATION="E:"+os.sep+"1Download1"+os.sep+"tst"
BOTTLENECK_LOCATION="E:"
#----------创建数据字典----------
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

#通过label_name和image_index读取图片地址
def get_image_path(image_list,label_name,image_index):
	dir_path=image_list[label_name]['label_path']
	iamge_name=image_list[label_name]['training'][image_index]
	path=os.path.join(dir_path,iamge_name)
	return path
	
	
#通过图片名称获取图片，并！计算（或读取）bottleneck！	
def get_or_cal_bottleneck(label_name,image_index):
	#由label 和 image_index获取图片
	image_path=get_iamge_path(label_name,image_index)#先获取地址
	iamge=[]#再由地址读取图片
	iamge=os.flie(iamge_path)
	
	#读取或计算bottle_neck
	#获取Inception-v3输入的bottleneck保存地址
	
	
	
#----------获得一个batch的输入数据----------
def get_random_cached_bottlenecks(num_classes,image_lists,batch_size):
	#先定义容器，盛放bottleneck数据
	bottleneck_training_batch=[]
	ground_true_batch=[]
	for n_time in range(batch_size):
		#先随机从字典里选择数据，生成lable编码，然后计算（或获取）产生的bottle值
		rand_label_index=random.randint（0,num_classes）
		label_name=image_lists.keys()[rand_label]
		ground_true=np.zero(num_classes,dtype=np.float32)
		ground_true_batch.append(ground_true)
		
		
		#找到该label下随机一条数据，计算其产生的bottle值
		image_index=random.randint(0,NUM_TRAINING)
		bottleneck=get_or_cal_bottleneck(，label_name,image_index)#重点是书写这个函数
		bottleneck_training_batch.append(bottleneck)
	return bottleneck_training_batch,ground_true_batch

	

	
	

def get_address_by_index(image_index,label,image_lists,_type):
	dir=image_lists[label]['label_path']
	image_name=image_list[label][_type][image_index]
	return address=os.path.join(dir,image_name)
	
	
	
	