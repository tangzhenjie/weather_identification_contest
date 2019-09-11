import tensorflow as tf
import os
import random
import sys
import pandas as pd
import cv2
import numpy as np
from data import get_statistic
dataset_dir = "../dataset/Train/"

def get_data(dataset):
    print("Loading training set...")
    filenames = dataset[:, 0]
    class_ids = [int(element) for element in dataset[:, 1]]
    data = []
    for index, filename in enumerate(filenames):
        image = cv2.imread(dataset_dir + filename, cv2.IMREAD_COLOR)                           # 读入一副彩色图像。图像的透明度会被忽略，(为什么透明度要被忽略，，可不可以换成)
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        # image = cv2.imread(filename, cv2.IMREAD_UNCHANGED)                     # 原始图片读入
        # image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)                       # 以灰色图片读入
        label = class_ids[index]
        data.append([image, label])
    random.seed(0)
    random.shuffle(data)
    print("Loading completed...")
    return data                                                                     # 格式是一个数组[[imgae, visit, label],[,,,],...]


def int64_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def image_to_tfexample(data, label):
    return tf.train.Example(features=tf.train.Features(feature={
        'data': bytes_feature(data),
        'label': int64_feature(label),
    }))


def _convert_dataset(data, tfrecord_path, dataset):
    """ Convert data to TFRecord format. """
    output_filename = os.path.join(tfrecord_path, dataset+".tfrecord")    # /home/wangdong/桌面/工程项目目录Spyter-tensorblow/研究生竞赛 /一带一路竞赛/初赛赛题/tfrecord/train.tfrecord
    tfrecord_writer = tf.python_io.TFRecordWriter(output_filename)        # 创建一个writer来写TFRecords文件
    length = len(data)                                                    # 三维数组的长度   84078
    for index, item in enumerate(data):
        data_ = item[0].tobytes()
        label = item[1]                                                   # 对应功能分类的标签
        example = image_to_tfexample(data_, label)
        tfrecord_writer.write(example.SerializeToString())                # 将样列序列化为字符串后， 写入out_filename文件中
        sys.stdout.write('\r>> Converting image %d/%d' % (index + 1, length))
        sys.stdout.flush()
    sys.stdout.write('\n')
    sys.stdout.flush()


if __name__ == '__main__':
    if not os.path.exists("../dataset/tfrecord/"):
        os.makedirs("../dataset/tfrecord/")

    train_datasets, val_datasets = get_statistic.getTrainAndValIndex()
    data = get_data(train_datasets)
    _convert_dataset(data, "../dataset/tfrecord/", "train")

    # data = get_data("train.txt")
    # _convert_dataset(data, "/home/wangdong/桌面/工程项目目录Spyter-tensorblow/研究生竞赛 /一带一路竞赛/自己优化版本/tfrecord/", "train")

    data = get_data(val_datasets)
    _convert_dataset(data, "../dataset/tfrecord/", "valid")