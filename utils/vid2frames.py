import os
import cv2
import argparse

__author__='Zhiyu Yin'

parser = argparse.ArgumentParser()
parser.add_argument('src_path')
parser.add_argument('dst_path')
args = parser.parse_args()

base_path_to_vids = args.src_path
base_path_to_frames = args.dst_path
if not os.path.exists(base_path_to_vids):
  raise ValueError(base_path_to_vids+'not exists')

num_vids = 0
for _, _, file_list in os.walk(base_path_to_vids):
  num_vids+=len(file_list)
print('%d videos found.'%num_vids)

count=1
for cur_location, dir_names, file_names in os.walk(base_path_to_vids):
  if not file_names:
    continue
  for vid_name in file_names:
    print('Extracting %dth vid'%count)
    path_to_vid = os.path.join(cur_location, vid_name)
    cap = cv2.VideoCapture(path_to_vid)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print('%d frames in total'%num_frames)
    for i in range(num_frames):
      ret, frame = cap.read()
      if ret == False:
        break
        #raise ValueError('fail to extract %dth frame'%(i+1) + vid_name)
      path_to_frame = os.path.join(base_path_to_frames, vid_name.split('.')[0])
      if not os.path.exists(path_to_frame):
        os.system('mkdir -p '+ path_to_frame)
      cv2.imwrite(os.path.join(path_to_frame, 'image-%06d.jpeg'%(i+1)), frame)
    count+=1
