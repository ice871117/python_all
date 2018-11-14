"""
Using CNN to mark audio piece tags among given ones.
"""

import tensorflow as tf
import time
import pandas as pd
import numpy as np
import random
import os

from inputs import Input
import model
import utils

# Parameters
WORKING_PATH = "../../../input/audio_train"
EXAM_PATH = "../../../input/audio_exam"
TEST_PATH = "../../../input/audio_test"
TRAIN_AND_EXAM_PATH = "../../../input/audio_train_and_exam"
TEMP_FILE = os.getcwd() + "\\temp.wav"
TRAIN_FILE = "train.csv"
SUBMIT_FILE = "sample_submission.csv"
TRAIN_DATA_SAVE_PATH = "ckpt/"
TRAIN_DATA_FILE_NAME = "kaggle_audio_tagging.ckpt"
START_FILE_INDEX = 0
REPEAT_SEGMENT_FOR_EACH_FILE = 4

FLATTEN_SIZE_W = 128  # log mel categories
FLATTEN_SIZE_H = 128

LEARNING_RATE = 1e-3
BATCH_SIZE = 32
DISPLAY_STEP = 10

# the time we skip for each wav file at head
SKIP_HEAD_IN_SECONDS = 0

# restore saver
RESTORE = False

# if true, just train, otherwise restore cache and skip training
TRAIN = True

# publish result or just tuning
PUBLISH = not TRAIN

# remember when we start
start_time = time.time()
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

sess = tf.InteractiveSession()
train_data_source = Input(WORKING_PATH,
                          TRAIN_FILE,
                          shuffle=True,
                          loop=True, temp_file_path=TEMP_FILE,
                          n_mfcc=FLATTEN_SIZE_W,
                          fixed_sample=FLATTEN_SIZE_H)
CATEGORY_NUM = train_data_source.get_category_num()
# tf Graph Input
x = tf.placeholder(tf.float32, [None, FLATTEN_SIZE_W, FLATTEN_SIZE_H])
y_ = tf.placeholder(tf.float32, [None, CATEGORY_NUM])
keep_prob = tf.placeholder(tf.float32)

x_audio = tf.reshape(x, [-1, FLATTEN_SIZE_W, FLATTEN_SIZE_H, 1])

# conv layer 1
conv_1 = model.conv2d_layer("cnn_1", x_audio, [5, 5], 1, 64, use_BN=True, training=TRAIN)
# conv layer 2
conv_2 = model.conv2d_layer("cnn_2", conv_1, [3, 3], 64, 128, pool_size=[2, 2], use_BN=True, training=TRAIN)
# conv layer 3
conv_3 = model.conv2d_layer("cnn_3", conv_2, [3, 3], 128, 128, pool_size=[2, 2], use_BN=True, training=TRAIN)

flat_layer, flat_size = model.get_cnn_layer_flat(conv_3)

# dense layer 1
dense1 = model.dense_layer("fc_1", flat_layer, flat_size, 1024, use_BN=True, use_dropout=True, local_keep_prob=keep_prob, activation=True, training=TRAIN)
# dense layer 2
dense2 = model.dense_layer("fc_2", dense1, 1024, 256, use_BN=True, use_dropout=True, local_keep_prob=keep_prob, activation=True, training=TRAIN)
# dense layer 3
dense3 = model.dense_layer("fc_3", dense2, 256, CATEGORY_NUM, training=TRAIN)

y_conv = tf.nn.softmax(dense3)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=dense3, labels=y_)
tf.summary.scalar('cross_entropy', cross_entropy)

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)

# prediction
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('accuracy', accuracy)

# for showing in tensor-board
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('./train', sess.graph)
exam_writer = tf.summary.FileWriter('./test')

# create saver
saver = tf.train.Saver(max_to_keep=1)
model_file = tf.train.latest_checkpoint(TRAIN_DATA_SAVE_PATH)
# model_file = tf.train.get_checkpoint_state(TRAIN_DATA_SAVE_PATH).model_checkpoint_path

# Initialize the variables (i.e. assign their default value)
print("perform first time initialization...")
tf.global_variables_initializer().run()

if RESTORE and model_file:
    saver.restore(sess, model_file)
    print("restore success model file=", model_file)
    for item in saver._var_list:
        print(item.name + " = ")
        print(sess.run(item.name))

speed_param = [-9, -6, -3, 3, 6, 9]
train_accuracy_avg = 0
max_accuracy = 0

if TRAIN:
    for i in range(2000):
        # rand_for_aug_param = random.randint(0, 100)
        # if rand_for_aug_param > 50:
        #     train_data_source.set_speed(speed_param[rand_for_aug_param % len(speed_param)])
        (batchX, batchY) = train_data_source.next(BATCH_SIZE)
        print("=====> one round complete: batchX shape=", batchX.shape, "; batchY shape=", batchY.shape)
        if i % DISPLAY_STEP == 0:
            train_cross_entropy, train_accuracy, summary = sess.run([cross_entropy, accuracy, merged],
                                                                    feed_dict={x: batchX, y_: batchY, keep_prob: 1.0})
            train_writer.add_summary(summary, i)
            print("step %d, training accuracy %f cross_entropy %f" % (i, train_accuracy, train_cross_entropy))
            train_accuracy_avg = utils.get_moving_avg(train_accuracy_avg, train_accuracy)
            if max_accuracy < train_accuracy_avg:
                max_accuracy = train_accuracy_avg
                print("saving model at accuracy_avg %f" % max_accuracy)
                saver.save(sess, TRAIN_DATA_SAVE_PATH + TRAIN_DATA_FILE_NAME, global_step=i + 1)
        sess.run(train_step, feed_dict={x: batchX, y_: batchY, keep_prob: 0.5})

train_writer.close()

if False:
    exam_cross_entropy_avg = 0
    exam_accuracy_avg = 0
    exam_data_source = Input(EXAM_PATH,
                             TRAIN_FILE, shuffle=True,
                             loop=False, temp_file_path=TEMP_FILE,
                             n_mfcc=FLATTEN_SIZE_W,
                             fixed_sample=FLATTEN_SIZE_H)
    for i in range(4000):
        if exam_data_source.is_eof():
            break
        # exam_data_source.set_current_index(random.randint(0, exam_data_source.totalFiles - BATCH_SIZE))
        testX, testY = exam_data_source.next(BATCH_SIZE)
        test_cross_entropy, test_accuracy, summary = sess.run([cross_entropy, accuracy, merged],
                                                              feed_dict={x: testX, y_: testY, keep_prob: 1.0})
        exam_cross_entropy_avg = utils.get_moving_avg(exam_cross_entropy_avg, test_cross_entropy)
        exam_accuracy_avg = utils.get_moving_avg(exam_accuracy_avg, test_accuracy)
        exam_writer.add_summary(summary, i)
        if i % DISPLAY_STEP == 0:
            print("step %d, exam accuracy %f cross_entropy %f" % (i, exam_accuracy_avg, exam_cross_entropy_avg))
    print("=====> test result training accuracy %f cross_entropy %f" % (exam_accuracy_avg, exam_cross_entropy_avg))
exam_writer.close()

if PUBLISH:
    # start publish:
    test_data_source = Input(TEST_PATH, TRAIN_FILE, shuffle=False,
                                  loop=False, temp_file_path=TEMP_FILE,
                                  n_mfcc=FLATTEN_SIZE_W,
                                  fixed_sample=FLATTEN_SIZE_H)
    result_frame = pd.DataFrame(np.zeros([test_data_source.get_total_files(), 2]), columns=['fname', 'label'])
    for i in range(0, test_data_source.get_total_files()):
        testX, _ = test_data_source.next(1)
        # testX = np.pad(testX, ((0, BATCH_SIZE - 1), (0, 0)), 'constant')
        logits = sess.run(y_conv,
                          feed_dict={x: testX, keep_prob: 1.0})
        result_frame.iloc[i, 0] = test_data_source.get_last_read_file_name()
        result_frame.iloc[i, 1] = utils.trans_index_to_label(test_data_source, utils.top_n_value(tf.squeeze(logits).eval(session=sess), 3))

    result_frame.to_csv(path_or_buf=SUBMIT_FILE)
    print("=====> publish done")

sess.close()
utils.print_duration(start_time)
print("===== job finish =====")
