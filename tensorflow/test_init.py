"""
A script to demonstrate usage of tf.zeros and tf.ones
"""
import numpy as np
import tensorflow as tf

sess = tf.InteractiveSession()
random_normal = tf.random_normal([2, 2])
print("random_normal===> \n", sess.run(random_normal))
truncated_normal = tf.truncated_normal([2, 2], stddev = 0.1)
print("truncated_normal===> \n", sess.run(truncated_normal))
constant = tf.constant(0.7, shape = [2, 3])
print("constant===> \n", sess.run(constant))