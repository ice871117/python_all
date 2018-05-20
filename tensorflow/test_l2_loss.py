import numpy as np
import tensorflow as tf

sess = tf.InteractiveSession()
input = np.float32(np.random.rand(4, 4))
print("input before loss===> \n", input)
result = tf.nn.l2_loss(input)
print("input===> \n", input)   #input doesn't change
print("l2_loss===> \n", sess.run(result))
