#coding:utf-8
import glob
import os
import os.path as osp
import random
import numpy as np
import cv2
import time
import tensorflow as tf

global cfg
global is_training
cfg, is_training=None, None

def _get_num_frames(vid_data,boxes):
  boxes=boxes.tolist()*int(vid_data.shape[0])
  box_ind=range(int(vid_data.shape[0]))
  return np.array(boxes).astype(np.float32), np.array(box_ind).astype(np.int32)

def _i3d_data_augment(vid_data,label):
  global cfg
  global is_training
  if cfg.INPUT.MODALITY == 'RGB':
    vid_data.set_shape(tf.TensorShape([None, 256, 340, 3]))
  else:
    vid_data.set_shape(tf.TensorShape([None, 256, 340, 2]))
  boxes=[]
  if is_training:
    y1=random.uniform(0.0, (256.0-224.0)/256.0)
    y2=y1+224.0/256.0
    x1=random.uniform(0.0, (340.0-224.0)/340.0)
    x2=x1+224.0/340.0
    boxes.append([y1,x1,y2,x2])
    boxes=boxes*64
    box_ind=range(64)
  else:
    print '###############################################'
    y1=(256.0-224.0)/(2.0*256.0)
    y2=y1+224.0/256.0
    x1=(340.0-224.0)/(2.0*340.0)
    x2=x1+224.0/340.0
    boxes.append([y1,x1,y2,x2])
    boxes,box_ind=tf.py_func(_get_num_frames,[vid_data,boxes],[tf.float32,tf.int32])

  return tf.image.crop_and_resize(vid_data,   # 64 * img_h * img_w * num_channels
                                  boxes,
                                  box_ind,
                                  crop_size=[224,224],
                                  method='bilinear') ,label


def _i3d_sample(vid):
  global cfg
  global is_training
  vid =str(vid)
  vid_path= osp.join(cfg.INPUT.DATA_DIR , vid)# 获取路径
  if cfg.INPUT.MODALITY == 'RGB':
    images = []
    rgb_frames= glob.glob(osp.join(vid_path,'img_*'))
    num_rgb_frames= len(rgb_frames)
    selections = range(num_rgb_frames)
    if is_training:
      # 从头开始，选取连续的64帧
      while num_rgb_frames<64:
        selections=selections*2
      selected = selections[0:64]
    else:
      # 从头开始，取所有帧
      selected = selections
    for i in selected:
      image=cv2.imread(rgb_frames[i])
      image=cv2.resize(image,(340,256))/255.0  # cv中是340指w，256指h
      images.append(image)
    images=np.array(images).astype(np.float32)
    return images
  
  else: # flow
    flows=[]
    flow_x_frames= glob.glob(osp.join(vid_path,'flow_x_*'))
    flow_y_frames= glob.glob(osp.join(vid_path,'flow_y_*'))
    assert len(flow_x_frames) == len(flow_y_frames)
    flow_frames= zip(flow_x_frames, flow_y_frames)
    num_flow_frames= len(flow_frames)
    selections = range(num_flow_frames)
    if is_training:
      # 从头开始，选取连续的64帧
      while num_rgb_frames<64:
        selections=selections*2
      selected = selections[0:64]
    else:
      # 从头开始，取所有帧
      selected = selections
    for i in selected:
      flow_x = cv2.imread(flow_sampled[i][0], cv2.IMREAD_GRAYSCALE)
      flow_x = cv2.resize(flow_x,(340,256))/255.0
      flow_y = cv2.imread(flow_sampled[i][1], cv2.IMREAD_GRAYSCALE)
      flow_y = cv2.resize(flow_y,(340,256))/255.0
      flow = np.dstack([flow_x, flow_y])
      flows.append(flow)
    flows=np.array(flows).astype(np.float32)
    return flows
    

def _get_data(vid,label):
  global cfg
  vid_data = tf.py_func(_i3d_sample,[vid],tf.float32)
  return vid_data, tf.one_hot(label,depth=cfg.NUM_CLASSES)

def _parse_split(split_path):
  with open(split_path,'r') as f:
    items=f.readlines()
    vids=[]
    labels=[]
    for i in range(len(items)):
      item=items[i].strip().split()
      vids.append(item[0].split('/')[-1])
      labels.append(int(item[2]))
  return vids, labels


def get_dataset_iter(config, isTraining=True):
  """
  读取数据，预处理，组成batch，返回
  """
  global cfg
  global is_training
  cfg = config
  is_training = isTraining

  if is_training:
    vids, labels = _parse_split(cfg.TRAIN.SPLIT_PATH)
    dataset = tf.data.Dataset.from_tensor_slices((vids,labels))
    dataset = dataset.map(_get_data, num_parallel_calls=1)  # 读取数据
    dataset = dataset.map(_i3d_data_augment, num_parallel_calls=1) # 数据增强
    # shuffle, get_batch
    dataset = dataset.repeat().shuffle(buffer_size=cfg.TRAIN.BATCH_SIZE*20).batch(cfg.TRAIN.BATCH_SIZE).prefetch(buffer_size=1)
  else:
    vids, labels = _parse_split(cfg.VALID.SPLIT_PATH)
    dataset = tf.data.Dataset.from_tensor_slices((vids,labels))
    dataset = dataset.map(_get_data, num_parallel_calls=1)
    dataset = dataset.map(_i3d_data_augment, num_parallel_calls=1)
    # shuffle, get_batch
    dataset = dataset.batch(cfg.VALID.BATCH_SIZE).prefetch(buffer_size=1)

  ite = dataset.make_one_shot_iterator()
  return ite
