"""
A script to demonstrate usage of tf.zeros and tf.ones
"""
import tensorflow as tf


sess = tf.InteractiveSession()
x = tf.ones([2, 3], tf.int32)
print(sess.run(x))
y = tf.zeros([10], tf.float32)
print(sess.run(y))
z = tf.random_uniform([3, 2], 1.0, 0)
print(sess.run(z))