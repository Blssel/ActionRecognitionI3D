#coding:UTF-8
import cv2
import numpy as np
import os
import pathlib

########
# 此文件需要在Y-npy文件夹下执行，同时所有生成目录也都在里面生成
# 源文件放在E:\\UCF-101-rand-croped下
########

SRC_DIR="E:"+os.sep+"UCF-101-rand-croped"
#DST_ROOT_DIR="E:"+os.sep+"Y-npy"

#遍历文件夹中的每一个视频，对其进行操作
class_count=0 
for cur_location,dir_names,file_names in os.walk(SRC_DIR):
	if not file_names:
		continue
	else:
		#获取视频
		class_count+=1
		print("It's processing the %dth class...\n"%class_count)
		video_count=0
		for file_name in file_names:
			#获取路径并打开视频
			video_path=os.path.join(cur_location,file_name)
			cap=cv2.VideoCapture(video_path)
			size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
			#获取每一帧，并进行处理
			ret,frame=cap.read()
			video_npy=np.array([],dtype=np.float32)
			flag=True
			while(ret==True):
				#转换为np.float32类型
				frame=np.float32(frame)
				#归一化到-1，1之间
				tmp=np.ones(frame.shape,dtype=np.float32)*127
				frame=(frame-tmp)/128.0
				#拼合
				if flag:
					video_npy=frame
					flag=False
					ret,frame=cap.read()
					continue
				video_npy=np.concatenate((video_npy,frame))
				#接着读下一帧
				ret,frame=cap.read()
			#reshape一下	
			num_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
			video_npy=video_npy.reshape(1,num_frames,size[0],size[1],3)
			print video_npy.shape
			
			#找到（或创建）文件夹并保存
			save_dir=os.path.basename(cur_location)
			if not os.path.exists(save_dir):
				os.mkdir(save_dir)
			np.save(pathlib.Path(save_dir+os.sep+file_name.split('.')[0]+".npy"),video_npy)
			
			video_count+=1
			print("%dth video:%s completed!\n\n\n"%(video_count,file_name))
			
			
			
			
			



