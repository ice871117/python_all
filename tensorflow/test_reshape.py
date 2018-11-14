import numpy as np
import tensorflow as tf

sess = tf.InteractiveSession()
input = np.float32(np.random.rand(4, 4))
result = tf.reshape(input, [2, -1])
print("input===> \n", input)   #input doesn't change
print("reshape===> \n", sess.run(result))
