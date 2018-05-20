'''
Using CNN to mark audio piece tags among given ones.
'''

import tensorflow as tf
import numpy as np
from datasource import DataSource
import time
import os

# Parameters
WORKING_PATH = "/Users/wiizhang/Documents/python_scripts/tensorflow/train_audio_category/downloaded_programs"
TEST_PATH = "/Users/wiizhang/Documents/python_scripts/tensorflow/train_audio_category/test_programs"
TEMP_PATH = "/Users/wiizhang/Documents/python_scripts/tensorflow/train_audio_category/temp.wav"
TRAIN_DATA_SAVE_PATH = "/Users/wiizhang/Documents/python_scripts/tensorflow/train_audio_category/"
TRAIN_DATA_FILE_NAME = "cnn_model"
CATEGORY_NUM = 10
START_FILE_INDEX = 2000
REPEAT_SEGMENT_FOR_EACH_FILE = 30
# almost 6 seconds of audio, we make it easier to be squared
# 262144 = 512 x 512
PROCESS_AUDIO_FRAMES = 16384
FLATTEN_SIZE = 128

LEARNING_RATE = 1e-8
BATCH_SIZE = 60
DISPLAY_STEP = 1

SKIP_HEAD_IN_SECONDS = 17 # the time we skip for each wav file at head


def weight_variable(shape):
    '''
    build up weight
    :param shape:
    :return:
    '''
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    '''
    build up bias
    :param shape:
    :return:
    '''
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

def conv2d(x, W):
    '''
    2 dimension convolution with step equals one
    :param x:
    :param W:
    :return:
    '''
    return tf.nn.conv2d(x, W, strides = [1,1,1,1], padding = 'SAME')

def max_pool(x, size = 2):
    '''
    size x size max pool
    :param x:
    :return:
    '''
    return tf.nn.max_pool(x, ksize = [1,size,size,1], strides = [1,size,size,1], padding = 'SAME')

def printDuration(startTime):
    duration = time.time() - start_time
    sec = int(duration + 0.5)
    min = int(sec / 60)
    sec = sec % 60
    hour = int(min / 60)
    min = min % 60
    print("time cost is %02d:%02d:%02d" % (hour, min, sec))



# remember when we start
start_time = time.time()

sess = tf.InteractiveSession()
dataSource = DataSource(WORKING_PATH, TEMP_PATH, CATEGORY_NUM, startIndex = START_FILE_INDEX, repeat = REPEAT_SEGMENT_FOR_EACH_FILE)
# tf Graph Input
x = tf.placeholder(tf.float32, [None, PROCESS_AUDIO_FRAMES])
# 0-9 digits recognition => 10 classes as we defined in CATEGORY_NUM
y_ = tf.placeholder(tf.float32, [None, CATEGORY_NUM])
x_audio = tf.reshape(x, [-1, FLATTEN_SIZE, FLATTEN_SIZE, 1])

# conv layer 1
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([1, 32])
h_conv1 = tf.nn.relu(tf.add(conv2d(x_audio, W_conv1),b_conv1))
h_pool1 = max_pool(h_conv1, size = 4)

# conv layer 2
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([1, 64])
h_conv2 = tf.nn.relu(tf.add(conv2d(h_pool1, W_conv2), b_conv2))
h_pool2 = max_pool(h_conv2, size = 4)

# conv layer 3
W_conv3 = weight_variable([3, 3, 64, 128])
b_conv3 = bias_variable([1, 128])
h_conv3 = tf.nn.relu(tf.add(conv2d(h_pool2, W_conv3), b_conv3))

# full connection layer 4
W_fc1 = weight_variable([8 * 8 * 128, 1024])
b_fc1 = bias_variable([1, 1024])
h_pool2_flat = tf.reshape(h_conv3, [-1, 8 * 8 * 128])
h_fc1 = tf.nn.relu(tf.add(tf.matmul(h_pool2_flat, W_fc1), b_fc1))

# drop out 1th
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# full connection layer 5
W_fc2 = weight_variable([1024, CATEGORY_NUM])
b_fc2 = bias_variable([1, CATEGORY_NUM])
y_conv = tf.nn.softmax(tf.add(tf.matmul(h_fc1_drop, W_fc2), b_fc2))

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y_conv, 1e-10, 1.0)), reduction_indices = [1]))
train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)

# prediction
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Initialize the variables (i.e. assign their default value)
tf.global_variables_initializer().run()
# create saver
saver = tf.train.Saver(max_to_keep = 1)
model_file = tf.train.latest_checkpoint(TRAIN_DATA_SAVE_PATH)
if model_file:
    saver.restore(sess, model_file)
# for i in range(10000):
#     if dataSource.isEOF():
#         break
#     (batchX, batchY) = dataSource.next(int(44100 * SKIP_HEAD_IN_SECONDS), BATCH_SIZE, PROCESS_AUDIO_FRAMES)
#     print("=====> one round complete: batchX shape=", batchX.shape, "; batchY shape=", batchY.shape)
#     if i % DISPLAY_STEP == 0:
#         train_cross_entropy = cross_entropy.eval(feed_dict = {x:batchX, y_:batchY, keep_prob:1.0})
#         train_accuracy = accuracy.eval(feed_dict = {x:batchX, y_:batchY, keep_prob:1.0})
#         print("step %d, training accuracy %f cross_entropy %f" % (i, train_accuracy, train_cross_entropy))
#         saver.save(sess, TRAIN_DATA_SAVE_PATH + TRAIN_DATA_FILE_NAME)
#     train_step.run(feed_dict = {x:batchX, y_:batchY, keep_prob:0.5})
# dataSource.close()

#start test:
testSource = DataSource(TEST_PATH, TEMP_PATH, CATEGORY_NUM, startIndex = 80, repeat = REPEAT_SEGMENT_FOR_EACH_FILE)
# test_cross_entropy = 0
# test_accuracy = 0
# loopSize = 150
# for i in range(0, loopSize):
#     (testX, testY) = testSource.next(int(44100 * SKIP_HEAD_IN_SECONDS), BATCH_SIZE, PROCESS_AUDIO_FRAMES)
#     test_cross_entropy += cross_entropy.eval(feed_dict = {x:testX, y_:testY, keep_prob:1.0})
#     test_accuracy += accuracy.eval(feed_dict = {x:testX, y_:testY, keep_prob:1.0})
# print("=====> test result training accuracy %f cross_entropy %f" % (test_accuracy / loopSize, test_cross_entropy / loopSize))

#test_single_file
def findMostPredict(arr):
    result = -1
    max = 0
    countMap = dict()
    for item in arr:
        if not countMap.get(item):
            countMap.setdefault(item, 1)
        else:
            countMap[item] += 1
    for (k,v) in countMap.items():
        if v > max:
            max = v
            result = k
    return result
for i in range(0, 10):
    (testX, testY) = testSource.next(int(44100 * SKIP_HEAD_IN_SECONDS), REPEAT_SEGMENT_FOR_EACH_FILE, PROCESS_AUDIO_FRAMES)
    test_one_hot = y_conv.eval(feed_dict = {x:testX, y_:testY, keep_prob:1.0})
    test_result = np.argmax(test_one_hot, 1)
    actual_result = np.argmax(testY, 1)
    print("=====> test single file, test_result=", findMostPredict(test_result)," of ", test_result, " actual_result", findMostPredict(actual_result), " of ", actual_result)
sess.close()
testSource.close()
printDuration(start_time)
