# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import pandas as pd
import random

BATCH_SIZE = 80
DISPLAY_STEP = 30
LEARNING_RATE = 0.01


################################  for processing data  #####################################


def normalization(arr):
    result = (arr - arr.mean()) / arr.std()
    return np.nan_to_num(result, False)


def process_string(arr):
    values = arr.values
    HASH_FOR_EMPTY = 0
    for i, item in enumerate(values):
        if not isinstance(item, str):
            values[i] = HASH_FOR_EMPTY
        else:
            values[i] = hash(item) / 1000.0
    return normalization(arr)


def split_string(dataframe, arr, *expansion_name):
    values = arr.values
    exp_size = len(expansion_name)
    HASH_FOR_EMPTY = 0
    for name in expansion_name:
        dataframe[name] = arr
    for i, item in enumerate(values):
        if not isinstance(item, str):
            for name in expansion_name:
                dataframe[name].values[i] = HASH_FOR_EMPTY
        else:
            #split for no more than expected times
            item_arr = item.split(' ', exp_size)
            item_arr_len = len(item_arr)
            for j in range(0, exp_size):
                if j < item_arr_len:
                    dataframe[expansion_name[j]].values[i] = hash(item_arr[j].strip(',.)('))
                else:
                    dataframe[expansion_name[j]].values[i] = HASH_FOR_EMPTY
    #normalize
    for name in expansion_name:
        dataframe[name] = normalization(dataframe[name])


def get_data(filePath, get_label):
    passengerInfo = pd.read_csv(filePath)
    data = None
    if get_label:
        data = passengerInfo.loc[0:, ['Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']]
    else:
        data = passengerInfo.loc[0:, ['Pclass', 'PassengerId', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']]
    data['Pclass'] = normalization(data['Pclass'])
    data['Sex'] = process_string(data['Sex'])
    split_string(data, data['Name'], 'Name0', 'Name1', 'Name2', 'Name3', 'Name4')
    split_string(data, data['Cabin'], 'Cabin0', 'Cabin1', 'Cabin2', 'Cabin3')
    split_string(data, data['Ticket'], 'Ticket0', 'Ticket1')
    data['Embarked'] = process_string(data['Embarked'])
    data['Age'] = normalization(data['Age'])
    data['SibSp'] = normalization(data['SibSp'])
    data['Parch'] = normalization(data['Parch'])
    data['Fare'] = normalization(data['Fare'])
    to_fetch = ['Pclass', 'Sex',
                'Age', 'SibSp', 'Fare',
                'Cabin0', 'Cabin1', 'Cabin2', 'Cabin3', 'Embarked']
    if not get_label:
        to_fetch.append('PassengerId')
    data = data.loc[0:, to_fetch]
    if get_label:
        return data, passengerInfo.loc[0:, ['Survived']]
    else:
        return data, None

def get_row_num(frame):
    return frame.iloc[:,0].size


training, labels = get_data('train.csv', True)

# print(training)

COLUMN_SIZE = training.columns.size

################################  for training  #####################################


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    pass
    # with tf.name_scope('summary'):
    #     mean = tf.reduce_mean(var)
    #     tf.summary.scalar('mean', mean)
    #     with tf.name_scope('stddev'):
    #         stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    #     tf.summary.scalar('stddev', stddev)
    #     tf.summary.scalar('max', tf.reduce_max(var))
    #     tf.summary.scalar('min', tf.reduce_min(var))
    #     tf.summary.histogram('histogram', var)


def weight_variable(shape, scope=''):
    '''
    build up weight
    :param shape:
    :return:
    '''
    with tf.name_scope(scope):
        initial = tf.truncated_normal(shape, stddev = 0.1)
        weights = tf.Variable(initial)
        # variable_summaries(weights)
    return weights


def bias_variable(shape, scope=''):
    '''
    build up bias
    :param shape:
    :return:
    '''
    with tf.name_scope(scope):
        initial = tf.constant(0.1, shape = shape)
        bias = tf.Variable(initial)
        # variable_summaries(bias)
    return bias


def add_layer(in_data, in_size, out_size, use_dropout = True, scope=''):
    with tf.name_scope(scope):
        W_fc1 = weight_variable([in_size, out_size], scope + '_w')
        b_fc1 = bias_variable([1, out_size], scope + '_b')
        h_fc1 = tf.nn.relu(tf.add(tf.matmul(in_data, W_fc1), b_fc1))
        if use_dropout:
            with tf.name_scope('dropout'):
                dropout = tf.nn.dropout(h_fc1, keep_prob)
            return dropout
        else:
            return h_fc1


sess = tf.InteractiveSession()
with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, COLUMN_SIZE])
    y_ = tf.placeholder(tf.float32, [None, 1])
keep_prob = tf.placeholder(tf.float32)

layer1 = add_layer(x, COLUMN_SIZE, 16, scope='lr_layer1')

W_fc_final = weight_variable([16, 1])
b_fc_final = bias_variable([1, 1])
with tf.name_scope('predict'):
    prediction = tf.sigmoid(tf.add(tf.matmul(layer1, W_fc_final), b_fc_final))
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_ - prediction), reduction_indices=[1]))
tf.summary.scalar('loss', loss)
with tf.name_scope('train_step'):
    train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)

merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('./train', sess.graph)
tf.global_variables_initializer().run()
i = 0
j = 0
train_row = get_row_num(training)
while j < 14000:
    end = i + BATCH_SIZE - 1 if i + BATCH_SIZE - 1 < train_row else train_row
    batchX = training.loc[i: end].values
    batchY = labels.loc[i: end].values
    if j % DISPLAY_STEP == 0:
        summary = merged.eval(feed_dict={x: batchX, y_: batchY, keep_prob: 1.0})
        train_writer.add_summary(summary, j)
        loss_value = loss.eval(feed_dict={x: batchX, y_: batchY, keep_prob: 1.0})
        prediction_value = prediction.eval(feed_dict={x: batchX, y_: batchY, keep_prob: 1.0})
        print("round (", j, ") loss = ", loss_value)
        # print("predict = ", prediction_value)
        # print("real = ", batchY)
    train_step.run(feed_dict={x: batchX, y_: batchY, keep_prob: 0.5})
    i = random.randint(0, 800 - BATCH_SIZE)
    j += 1

test, labels = get_data('test00.csv', True)
test_row = get_row_num(test)
result = np.zeros([test_row, 1])
train_writer.close()

i = 0
while i < test_row:
    end = i + BATCH_SIZE - 1 if i + BATCH_SIZE - 1 < test_row else test_row - 1
    batchX = training.loc[i: end].values
    prediction_value = prediction.eval(feed_dict={x: batchX, keep_prob: 1.0})
    result[i : end + 1] = prediction_value
    i = end + 1

result = np.apply_along_axis(lambda x: 1 if x >= 0.5 else 0, 1, result)
result.resize(test_row, 1)

test_loss = np.mean(np.sum(np.square(result - labels), axis=1)) # using MSE to estimate loss
label_value = labels.values
for i in range(0, result.shape[0]):
    print("result = ", result[i][0], "; ", label_value[i][0])
print("test_loss = ", test_loss)

# test['Survived'] = result
# submission = test.loc[0:, ['PassengerId', 'Survived']]
# submission.to_csv('gender_submission.csv')
# print(result)
# print(submission)