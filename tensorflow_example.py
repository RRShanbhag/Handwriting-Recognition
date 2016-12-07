from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

# sess = tf.InteractiveSession()

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

train = mnist.train

print(train)
