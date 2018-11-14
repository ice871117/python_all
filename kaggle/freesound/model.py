import tensorflow as tf


def variable_summaries(var, name):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope(name + '_summaries'):
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
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)


def bias_variable(shape):
    """
    build up bias
    :param shape:
    :return:
    """
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    """
    2 dimension convolution with step equals one
    :param x:
    :param W:
    :return:
    """
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool(x, size):
    """
    size x size max pool
    :param x:
    :param size:
    :return:
    """
    return tf.nn.max_pool(x, ksize=[1, size[0], size[1], 1], strides=[1, size[0], size[1], 1], padding='SAME')


def conv2d_layer(layer_name, data, kernel, in_channels, out_channels, pool_size=None,
                  use_BN=False, use_dropout=False, local_keep_prob=0, training=False):
    W_conv = weight_variable([kernel[0], kernel[1], in_channels, out_channels])
    variable_summaries(W_conv, layer_name + "_weights")
    b_conv = bias_variable([1, out_channels])
    variable_summaries(b_conv, layer_name + "_biases")
    conv_res = tf.add(conv2d(data, W_conv), b_conv)
    if use_BN:
        conv_res = tf.layers.batch_normalization(conv_res, training=training, epsilon=1e-5)
    if use_dropout:
        conv_res = tf.nn.dropout(conv_res, local_keep_prob)
    h_conv = tf.nn.relu(conv_res)
    tf.summary.histogram(layer_name + '_Wx_plus_b_activations', h_conv)
    if pool_size:
        h_conv = max_pool(h_conv, pool_size)
    return h_conv


def dense_layer(layer_name, input, in_size, out_size, use_BN=False, use_dropout=False, local_keep_prob=0, activition=False, training=False):
    W_fc1 = weight_variable([in_size, out_size])
    variable_summaries(W_fc1, layer_name + "_weight")
    b_fc1 = bias_variable([1, out_size])
    variable_summaries(b_fc1, layer_name + "_bias")
    fc_res1 = tf.add(tf.matmul(input, W_fc1), b_fc1)
    if use_BN:
        fc_res1 = tf.layers.batch_normalization(fc_res1, training=training, epsilon=1e-5)
    if use_dropout:
        # dropout
        fc_res1 = tf.nn.dropout(fc_res1, local_keep_prob)
    if activition:
        fc_res1 = tf.nn.relu(fc_res1)
    return fc_res1


def get_cnn_layer_flat(layer):
    shape = layer.shape
    flat = tf.reshape(layer, [-1, shape[1] * shape[2] * shape[3]])
    return flat, flat.get_shape().as_list()[1]