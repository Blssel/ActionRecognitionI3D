# -*- coding: utf-8 -*-
#!/usr/bin/env python
import glob
import json
import os.path
import random
import numpy as np
#import tensorflow as tf
#from tensorflow.python.platform import gfile


CACHE_DIR = './'
#INPUT_DATA = '/extra_store/Y-npy/'  #目录修改
INPUT_DATA='/extra_disk/dataset/hmdb_rand_crop_npy/'

#验证和测试的百分比
VALIDATION_PERCENTAGE = 10
TEST_PERCENTAGE = 10

BATCH = 100

# 读取所有的数据并按测试、验证、训练分开
def create_image_lists(testing_percentage, validation_percentage):
    # 得到的所有视频储存在result这个字典里
    result = {}
    sub_dirs = [x[0] for x in os.walk(INPUT_DATA)]  # os.walk 通过在目录中游走输出在目录中的文件名
    # 由于os.walk()列表第一个是'./'，即得到的第一个目录是当前目录，所以排除

    # 遍历各个label文件夹
    is_root_dir = True
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue

        # 遍历当前目录下所有有效视频
        extensions = ['npy','NPY']
        file_list = []  # 给所有的文件建立一个文件列表
        dir_name = os.path.basename(sub_dir)  # os.path.basename 提取参数path的最后一部分
        for extension in extensions:
            file_glob = os.path.join(INPUT_DATA, dir_name, '*.' + extension)  # 连接两个文件名地址
            file_list.extend(glob.glob(file_glob))  # glob()返回匹配指定模式的文件名或目录
        if not file_list: continue

        label_name = dir_name.lower()  # 生成label，实际就是小写文件夹名

        # 初始化
        training_images = []
        testing_images = []
        validation_images = []
        for file_name in file_list:
            base_name = os.path.basename(file_name)  # 去路径，只保留文件名

            # 随机划分数据
            chance = np.random.randint(100)
            if chance < validation_percentage:
                validation_images.append(base_name)
            elif chance < (testing_percentage + validation_percentage):
                testing_images.append(base_name)
            else:
                training_images.append(base_name)
        # 本标签字典项生成
        result[label_name] = {
            'dir': dir_name,
            'training': training_images,
            'testing': testing_images,
            'validation': validation_images,
        }
    return result

# 根据类别名称、所属数据集和图片编号获得一张图片的地址
# image_list 给出了所有图片信息，index给定了需要获取的图片的编号（随机数索引），category指定了图片是在数据集、验证集还是测试集
def get_image_path(image_lists, image_dir, label_name, index, category):
    label_lists = image_lists[label_name]  # 获取给定类别中所有图片信息
    category_list = label_lists[category]  # 获取目标category图片列表
    mod_index = index % len(category_list)  # 随机获取一张图片的索引
    base_name = category_list[mod_index]  # 通过索引获取图片名
    sub_dir = label_lists['dir']
    full_path = os.path.join(image_dir, sub_dir, base_name)  # 最终地址为数据根目录加类别的文件夹加图片的名称，image_dir: 外层文件夹（内部是标签文件夹）
    return full_path

def main():
    # 读取所有图片，生成文件字典
    image_lists = create_image_lists(TEST_PERCENTAGE, VALIDATION_PERCENTAGE)
    #print (image_lists)
    json.dump(image_lists, open('hmdb_rgb_list.json', 'w'))
    with open('hmdb_rgb_list.json', 'r') as f:
        data = json.load(f)
    print(data)

if __name__ == '__main__':
    main()
