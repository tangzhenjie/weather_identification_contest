from data import DataGenerator
from network import ResNet

import tensorflow as tf
import datetime
import numpy as np
import math
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
slim = tf.contrib.slim

LOG_DIR = './log/'
CHECKPOINT_DIR = './checkpoint/'
NUM_CLASSES = 9
BATCHSIZE = 40
LEARNINT_RATE = 0.0001
EPOCHS = 100
weight_path = "./resnet_first_wights/"

def get_loss(output_concat, onehot):
    with tf.name_scope("loss"):
        # cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_predict, labels=batch_y)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=output_concat, labels=onehot)
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        loss = tf.add_n([cross_entropy_mean] + regularization_losses)
        tf.summary.scalar('train_loss', loss)
    return loss


##################### get the input pipline ############################
DataGenerator = DataGenerator.DataGenerator()
TrainDataset = DataGenerator.get_batch(BATCHSIZE, tag="training")
EvalDataset = DataGenerator.get_batch(BATCHSIZE, tag="evaling")

# get the dataset statistics
trainset_length = 5113 # 换
eval_set_length = 1274 # 换
print("train_set_length:%d" % trainset_length)
print("eval_set_length:%d" % eval_set_length)


iterator = tf.data.Iterator.from_structure(TrainDataset.output_types, TrainDataset.output_shapes)
next_batch = iterator.get_next()
training_init_op = iterator.make_initializer(TrainDataset)
validation_init_op = iterator.make_initializer(EvalDataset)
##################### get the input pipline ############################


##################### setup the network ################################
x = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))
y = tf.placeholder(tf.int32, shape=(None, NUM_CLASSES))
is_training = tf.placeholder('bool', [])
keep_prob = tf.placeholder(tf.float32)


depth = 50    # 可以是50、101、152
ResNetModel = ResNet.ResNetModel(is_training, depth, NUM_CLASSES)
fc_image = ResNetModel.inference(x)
net_output = ResNet.get_net_output(fc_image=fc_image, classNum=NUM_CLASSES, KEEP_PROB=keep_prob)
prediction = tf.argmax(net_output, 1)
groundtruth = tf.argmax(y, 1)

# 训练操作
with tf.name_scope("train"):
    loss = get_loss(net_output, y)
    train_layers = ["scale5", "fc"]
    train_op = ResNetModel.optimize(loss=loss, learning_rate=LEARNINT_RATE, train_layers=train_layers)
# 评价操作
with tf.name_scope("eval"):
    correct_pred = tf.equal(tf.argmax(net_output, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    tf.summary.scalar('train_accuracy', accuracy)
summary_op = tf.summary.merge_all()
# 混淆矩阵
confus_matrix = tf.confusion_matrix(tf.argmax(y, 1), tf.argmax(net_output, 1), num_classes=NUM_CLASSES, name="con_matrix")
##################### setup the network ################################
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    # initial variables
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    # 获取预训练的权重
    ResNetModel.load_original_weights(weight_path=weight_path, session=sess)
    # 判断有没有checkpoint

    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored .....")
    sess.graph.finalize()
    # summary
    train_writer = tf.summary.FileWriter(LOG_DIR + "/train")
    eval_writer = tf.summary.FileWriter(LOG_DIR + "/eval")

    # 训练过程
    print("training start")
    train_batches_of_epoch = int(math.ceil(trainset_length/BATCHSIZE))
    for epoch in range(EPOCHS):
        sess.run(training_init_op)
        print("{} Epoch number: {}".format(datetime.datetime.now(), epoch + 1))
        step = 1
        while step <= train_batches_of_epoch:
            img_batch, label_batch = sess.run(next_batch)
            pre, true, _, loss_value, merge, accu = sess.run([prediction, groundtruth, train_op, loss, summary_op, accuracy], feed_dict={x: img_batch, y: label_batch, is_training: True, keep_prob:0.5})
            if step % 20 == 0:
                print("{} {} loss = {:.4f}".format(datetime.datetime.now(), step, loss_value))
                print("accuracy{}".format(accu))
                print(pre)
                print(true)
                saver.save(sess, CHECKPOINT_DIR + "model.ckpt", step)
                train_writer.add_summary(merge, epoch * train_batches_of_epoch + step)
                print("checkpoint saved")
            step = step + 1


        # 验证过程
        print("{} Start validation".format(datetime.datetime.now()))
        test_acc = 0.0
        test_count = 0
        eval_batches_of_epoch = int(math.ceil(eval_set_length/BATCHSIZE))

        sess.run(validation_init_op)
        con_mat = np.ones((NUM_CLASSES, NUM_CLASSES), dtype=np.int32)
        for tag in range(eval_batches_of_epoch):
            img_batch, label_batch = sess.run(next_batch)
            pre, true, acc, con_matrix = sess.run([prediction, groundtruth, accuracy,  confus_matrix], feed_dict={x: img_batch, y: label_batch, is_training: False, keep_prob:1.0})
            con_mat = con_mat + con_matrix
            test_acc += acc
            test_count += 1
            #print("the {} time Validation Accuracy = {:.4f}".format(tag, acc))
            #print(pre)
            #print(true)
            #print(con_mat)
        test_acc /= test_count
        s = tf.Summary(value=[
            tf.Summary.Value(tag="validation_accuracy", simple_value=test_acc)
        ])
        eval_writer.add_summary(s, epoch + 1)
        print("{} Validation Accuracy = {:.4f}".format(datetime.datetime.now(), test_acc))

