import numpy as np
import tensorflow as tf

sess = tf.InteractiveSession()
input = np.float32(np.random.randint(0, 2, size=(4, 4)))
result = tf.nn.softmax(input)
print("input===> \n", input)   #input doesn't change
print("softmax===> \n", sess.run(result))
