"""
Using CNN to mark audio piece tags among given ones.
"""

import tensorflow as tf
from audio_datasource import DataSource
import time
import pandas as pd
import numpy as np

# Parameters
WORKING_PATH = "../../../input/audio_train"
EXAM_PATH = "../../../input/audio_exam"
TEST_PATH = "../../../input/audio_test"
TRAIN_FILE = "train.csv"
SUBMIT_FILE = "sample_submission.csv"
TRAIN_DATA_SAVE_PATH = "./cached_model"
TRAIN_DATA_FILE_NAME = "kaggle_audio_tagging"
START_FILE_INDEX = 0
REPEAT_SEGMENT_FOR_EACH_FILE = 10
# almost 6 seconds of audio, we make it easier to be squared
# 262144 = 512 x 512
# 16384 = 128 x 128
PROCESS_AUDIO_FRAMES = 16384
FLATTEN_SIZE = 128

LEARNING_RATE = 1e-4
BATCH_SIZE = 6 * REPEAT_SEGMENT_FOR_EACH_FILE
DISPLAY_STEP = 10

# the time we skip for each wav file at head
SKIP_HEAD_IN_SECONDS = 0
# publish result or just tuning
PUBLISH = False


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def weight_variable(shape):
    """
    build up weight
    :param shape:
    :return:
    """
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """
    build up bias
    :param shape:
    :return:
    """
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)


def conv2d(x, W):
    """
    2 dimension convolution with step equals one
    :param x:
    :param W:
    :return:
    """
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool(x, size = 2):
    """
    size x size max pool
    :param x:
    :return:
    """
    return tf.nn.max_pool(x, ksize=[1, size, size, 1], strides=[1, size, size, 1], padding='SAME')


def print_duration(before):
    duration = time.time() - before
    sec = int(duration + 0.5)
    min = int(sec / 60)
    sec = sec % 60
    hour = int(min / 60)
    min = min % 60
    print("time cost is %02d:%02d:%02d" % (hour, min, sec))


def add_cnn_layer(layer_name, data, kernel_w, kernel_h, in_channels, out_channels, pool_size=0):
    with tf.name_scope(layer_name):
        with tf.name_scope("weights"):
            W_conv = weight_variable([kernel_w, kernel_h, in_channels, out_channels])
            variable_summaries(W_conv)
        with tf.name_scope("biases"):
            b_conv = bias_variable([1, out_channels])
            variable_summaries(b_conv)
        with tf.name_scope('Wx_plus_b'):
            h_conv = tf.nn.relu(tf.add(conv2d(data, W_conv), b_conv))
            tf.summary.histogram('activations', h_conv)
        if pool_size > 0:
            with tf.name_scope("max_pool"):
                return max_pool(h_conv, size=pool_size)
        return h_conv


def get_cnn_layer_flat(layer):
    shape = layer.shape
    flat = tf.reshape(layer, [-1, shape[1] * shape[2] * shape[3]])
    return flat, flat.get_shape().as_list()[1]


def get_moving_avg(old_value, new_value, ratio):
    if old_value > 0:
        return ratio * old_value + (1 - ratio) * new_value
    else:
        return new_value


def top_n_value(arr, n):
    sorted_dict = dict()
    for index, item in enumerate(arr):
        sorted_dict[item] = index
    sorted_keys = sorted(sorted_dict.keys(), reverse=True)
    result = []
    for index, key in enumerate(sorted_keys):
        if index >= n:
            break
        result.append(sorted_dict.get(key))
    return result


def trans_index_to_label(data_source, arr):
    if len(arr) > 0:
        res_str = ""
        for i in arr:
            if i > 0:
                res_str += " "
            res_str += data_source.find_label_by_index(i)
        print("top_n_value=", res_str)
        return res_str
    else:
        return ""


# remember when we start
start_time = time.time()

sess = tf.InteractiveSession()
train_data_source = DataSource(WORKING_PATH, TRAIN_FILE, start_index= START_FILE_INDEX, repeat = REPEAT_SEGMENT_FOR_EACH_FILE)
CATEGORY_NUM = train_data_source.get_category_num()
# tf Graph Input
x = tf.placeholder(tf.float32, [None, PROCESS_AUDIO_FRAMES])
y_ = tf.placeholder(tf.float32, [None, CATEGORY_NUM])
x_audio = tf.reshape(x, [-1, FLATTEN_SIZE, FLATTEN_SIZE, 1])

# conv layer 1
conv_1 = add_cnn_layer("cnn_1", x_audio, 5, 5, 1, 32, pool_size=4)

# conv layer 2
conv_2 = add_cnn_layer("cnn_2", conv_1, 5, 5, 32, 64, pool_size=4)

# conv layer 3
conv_3 = add_cnn_layer("cnn_3", conv_2, 3, 3, 64, 128, pool_size=0)

flat_layer, flat_size = get_cnn_layer_flat(conv_3)

# full connection layer 1
W_fc1 = weight_variable([flat_size, 1024])
b_fc1 = bias_variable([1, 1024])
h_fc1 = tf.nn.relu(tf.add(tf.matmul(flat_layer, W_fc1), b_fc1))

# drop out
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# full connection layer 2
W_fc2 = weight_variable([1024, CATEGORY_NUM])
b_fc2 = bias_variable([1, CATEGORY_NUM])
y_conv = tf.nn.softmax(tf.add(tf.matmul(h_fc1_drop, W_fc2), b_fc2))

with tf.name_scope("cross_entropy"):
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y_conv, 1e-10, 1.0)), reduction_indices=[1]))
tf.summary.scalar('cross_entropy', cross_entropy)
with tf.name_scope("train"):
    train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)

# prediction
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
with tf.name_scope("accuracy"):
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('accuracy', accuracy)

# for showing in tensor-board
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('./train', sess.graph)
test_writer = tf.summary.FileWriter('./test')

# Initialize the variables (i.e. assign their default value)
tf.global_variables_initializer().run()
# create saver
saver = tf.train.Saver(max_to_keep=1)
model_file = tf.train.latest_checkpoint(TRAIN_DATA_SAVE_PATH)
if model_file:
    saver.restore(sess, model_file)
for i in range(10000):
    if train_data_source.isEOF():
        break
    (batchX, batchY) = train_data_source.next(int(44100 * SKIP_HEAD_IN_SECONDS), BATCH_SIZE, PROCESS_AUDIO_FRAMES)
    print("=====> one round complete: batchX shape=", batchX.shape, "; batchY shape=", batchY.shape)
    if i % DISPLAY_STEP == 0:
        train_cross_entropy, train_accuracy, summary = sess.run([cross_entropy, accuracy, merged], feed_dict={x: batchX, y_: batchY, keep_prob: 1.0})
        train_writer.add_summary(summary, i)
        print("step %d, training accuracy %f cross_entropy %f" % (i, train_accuracy, train_cross_entropy))
        saver.save(sess, TRAIN_DATA_SAVE_PATH + TRAIN_DATA_FILE_NAME)
    train_step.run(feed_dict={x: batchX, y_: batchY, keep_prob: 0.5})

train_writer.close()

if not PUBLISH:
    # start exam:
    exam_data_source = DataSource(TEST_PATH, TRAIN_FILE, start_index=0, repeat=REPEAT_SEGMENT_FOR_EACH_FILE)
    test_cross_entropy_avg = 0
    test_accuracy_avg = 0
    MOVING_AVG_RATIO = 0.98
    for i in range(0, 10000):
        if train_data_source.isEOF():
            break
        testX, _ = exam_data_source.next(int(44100 * SKIP_HEAD_IN_SECONDS), BATCH_SIZE, PROCESS_AUDIO_FRAMES)
        test_cross_entropy, test_accuracy, summary = sess.run([cross_entropy, accuracy, merged], feed_dict={x: testX, keep_prob: 1.0})
        test_cross_entropy_avg = get_moving_avg(test_cross_entropy_avg, test_cross_entropy, MOVING_AVG_RATIO)
        test_accuracy_avg = get_moving_avg(test_accuracy_avg, test_accuracy, MOVING_AVG_RATIO)
        test_writer.add_summary(summary, i)

    test_writer.close()
    print("=====> test result training accuracy %f cross_entropy %f" % (test_accuracy_avg, test_cross_entropy_avg))

if PUBLISH:
    # start publish:
    test_data_source = DataSource(TEST_PATH, TRAIN_FILE, start_index=0, repeat=REPEAT_SEGMENT_FOR_EACH_FILE)
    result_frame = pd.DataFrame(np.zeros([test_data_source.totalFiles, 2]), columns=['fname', 'label'])
    for i in range(0, test_data_source.totalFiles):
        testX, _ = test_data_source.next(int(44100 * SKIP_HEAD_IN_SECONDS), REPEAT_SEGMENT_FOR_EACH_FILE, PROCESS_AUDIO_FRAMES)
        logits = y_conv.eval(feed_dict={x: testX, keep_prob: 1.0})
        result_frame.iloc[i, 0] = test_data_source.get_last_read_file_name()
        result_frame.iloc[i, 1] = trans_index_to_label(test_data_source, top_n_value(logits.sum(axis=0), 3))

    result_frame.to_csv(path_or_buf=SUBMIT_FILE)
    print("=====> publish done")

sess.close()
print_duration(start_time)
print("===== job finish =====")
