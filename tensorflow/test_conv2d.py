"""
A script to demonstrate usage of tf.zeros and tf.ones
"""
import numpy as np
import tensorflow as tf

sess = tf.InteractiveSession()
input = tf.random_normal([1,4,4,3])
print("input===> \n", sess.run(input))
filter = tf.random_normal([2,2,3,1])
print("filter===> \n", sess.run(filter))
op = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='VALID')
print("conv2d result===> \n", sess.run(op))