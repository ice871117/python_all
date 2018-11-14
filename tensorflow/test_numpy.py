"""
A script to demonstrate usage of tf.zeros and tf.ones
"""
import numpy as np

x = np.float32(np.random.rand(3, 2))
print(x)
print("--------------------------------------")
y = np.dot([0.1, 0.2, 0.5], x) + 0.3
print(y)