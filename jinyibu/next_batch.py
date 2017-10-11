#!/usr/bin/python2
#coding:UTF-8

#从训练集里随机获取一个batch的数据
#假如有BATCH的大小为BATCH_SIZE。则每次随机在训练集里选择一个类，然后随机选择一个数据
#next_batch函数获得的batch的最终形式为张量形式
#返回是一个batch的bottleneck和对应groundtruth

def get_random_batch_bottleneck(sess, n_classes, video_lists, how_many, category, jpeg_data_tensor, bottleneck_tensor):
	#先定义好要返回的两个主角，bottleneck和groundtruth
	bottlenecks = []
	ground_truths = []
	for _ in range(how_many):#循环BATCH_SIZE次
		label_index = random.randrange(n_classes)#随机选一类
		label_name = list(video_lists.keys())[label_index]#获得这个类名
		video_index = random.randrange(65536)#？？？？？？？？为什么是65536


		#计算bottleneck的值(待看！！！！！！！！)
		bottleneck = get_or_create_bottleneck(sess, video_lists, label_name, video_index, category, jpeg_data_tensor, bottleneck_tensor)
		
		#生成这个类对应的groundtruth编码（维度就是类别数）
		ground_truth = np.zeros(n_classes, dtype=np.float32)#数据类型是float32型
		ground_truth[label_index] = 1.0#让这个类的索引对应位的值取1

		#bottleneck和groundtruth并入batch中
		bottlenecks.append(bottleneck)
		ground_truths.append(ground_truth)
		
	#愉快地返回
	return bottlenecks , ground_truth



		

def get_or_create_bottleneck(sess, image_lists, label_name, index, category, jpeg_data_tensor, bottleneck_tensor):
	#获取选中类别下的所有数据元信息
	label_lists = video_lists[label_name]
	
	#去缓存中寻找该类
	sub_dir = label_lists['dir']#获取该标签类所在路径
	sub_dir_path = os.path.join(CACHE_DIR, sub_dir)#推导bottleneck的保存文件夹路径
	#如果不存在就创建该文件夹
	if not os.path.exists(sub_dir_path): os.makedirs(sub_dir_path)
	#计算bottleneck的全路径（！！！！！！！！）
	bottleneck_path = get_bottleneck_path(video_lists, label_name, index, category)#参数~！！！！！！！

	#如果未缓存则计算并缓存
	if not os.path.exists(bottleneck_path):
		#读取视频，以备运算用(！！！！！！)
		video_path = get_video_path(video_lists, INPUT_DATA, label_name, index, category)
		npy_data = np.load(libpath.Path(video_path))#必须修改为读取npy的方式！！！！！！！！！！！！！！！

		#计算bottleneck(待看和待改！！！！！！！！！)
		bottleneck_values = run_bottleneck_on_video(sess, npy_data, jpeg_data_tensor, bottleneck_tensor)#参数！！！！！！

		#把bottleneck保存下来！！！！！！！！！！必须修改存成npy格式(注意后缀)
		np.save(pathlib.Path(bottleneck_path),npy_data)
	#否则直接读取即可(也是需要从npy文件中读取!!!!!!)
	else:
		bottleneck_values=np.load(pathlib.Path(bottleneck_path))

	#返回bottleneck_value
	return bottleneck_values


def get_bottleneck_path(video_lists, label_name, index, category):
	return get_video_path(video_lists, CACHE_DIR, label_name, index, category) + '.npy'


def get_video_path(video_lists, video_dir, label_name, index, category):
	label_lists = video_lists[label_name]
	category_list = label_lists[category]
	mod_index = index % len(category_list)
	base_name = category_list[mod_index]
	sub_dir = label_lists['dir']
	full_path = os.path.join(image_dir, sub_dir, base_name)
	return full_path


def run_bottleneck_on_image(sess, image_data, image_data_tensor, bottleneck_tensor):

	bottleneck_values = sess.run(bottleneck_tensor, {image_data_tensor: image_data})

	bottleneck_values = np.squeeze(bottleneck_values)
	
	return bottleneck_values
