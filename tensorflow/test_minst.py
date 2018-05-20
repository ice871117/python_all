"""
A script to demonstrate usage of tf.zeros and tf.ones
"""
import numpy as np
import tensorflow as tf
import input_data

minst = input_data.read_data_sets("MINST_data/", one_hot=True)

presetB = np.float32(np.random.rand(784, 2))

print("presetB:")
print(presetB)

x = tf.placeholder("float", [None, 784])
y_ = tf.placeholder("float", [None, 2])

W = tf.Variable(tf.random_uniform([784, 2], -1.0, 1.0))
b = tf.Variable(tf.zeros([2]))
y = tf.matmul(x, W) + b

# 最小化方差
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# 初始化变量
init = tf.initialize_all_variables()

# 启动图 (graph)
sess = tf.Session()
sess.run(init)

temp_ys = np.arange(200).reshape(100, 2)
# 拟合平面
for _ in range(1000):
    batch_xs, batch_ys = minst.train.next_batch(100)
    j = 0
    for img in batch_xs:
        temp_ys[j] = np.dot(img, presetB) + [0.2, 0.3]
        j += 1
    sess.run(train_step, feed_dict={x: batch_xs, y_: temp_ys})
print("W:")
print(sess.run(W))
print("b:")
print(sess.run(b))