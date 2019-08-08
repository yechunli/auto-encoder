import tensorflow as tf
import numpy as np
import os
from tensorflow.examples.tutorials.mnist import input_data
data_dir = 'F:\python_project\mnist'
mnist = input_data.read_data_sets(data_dir, one_hot=True)
class NN():
    def __init__(self):
        self.data = tf.placeholder(dtype=tf.float32, shape=[None, 784])
        self.label = tf.placeholder(dtype=tf.float32, shape=[None, 10])
        self.model()
    def initializ(self, symbol, layers, code, name):
        layer = [784, 500, 150]
        if code == 'encode':
            if symbol == 'w':
                parameter = tf.Variable(tf.truncated_normal(shape=[layer[layers-1], layer[layers]],dtype=tf.float32, stddev=0.1), name=name)
            elif symbol == 'b':
                parameter = tf.Variable(tf.zeros(shape=[layer[layers]]), dtype=tf.float32)
        elif code == 'decode':
            if symbol == 'w':
                parameter = tf.Variable(tf.truncated_normal(shape=[layer[3-layers], layer[2-layers]],dtype=tf.float32, stddev=0.1))
            elif symbol == 'b':
                parameter = tf.Variable(tf.zeros(shape=[layer[2-layers]]))
        return parameter
    def model(self):
        w1 = self.initializ('w', 1, 'encode', 'w1')
        b1 = self.initializ('b', 1, 'encode', 'b1')
        w2 = self.initializ('w', 2, 'encode', 'w2')
        b2 = self.initializ('b', 2, 'encode', 'b2')
        w3 = self.initializ('w', 1, 'decode', 'w3')
        b3 = self.initializ('b', 1, 'decode', 'b3')
        w4 = self.initializ('w', 2, 'decode', 'w4')
        b4 = self.initializ('b', 2, 'decode', 'b4')
        self.w1 = w1
        self.w2 = w2
        hidden1 = tf.nn.relu(tf.add(tf.matmul(self.data, w1), b1))
        hidden2 = tf.nn.relu(tf.add(tf.matmul(hidden1, w2),b2))
        hidden3 = (tf.add(tf.matmul(hidden2, w3), b3))
        hidden3 = tf.nn.relu(tf.add(tf.matmul(hidden2, w3), b3))
        output = tf.add(tf.matmul(hidden1, w4), b4)
        #w5 = tf.Variable(tf.truncated_normal(shape=[150, 10], dtype=tf.float32, stddev=0.1), name='w5')
        #b5 = tf.Variable(tf.zeros(shape=[10], dtype=tf.float32))
        #self.w5 = w5
        #output = tf.add(tf.matmul(hidden2, w5), b5, name='ww')
        self.output = output
        l2_loss = tf.nn.l2_loss(w1)+tf.nn.l2_loss(w2)
        
        #self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.label, logits=output))
        #l2_loss = tf.nn.l2_loss(w2)
        #self.loss = tf.reduce_mean((output - hidden1) ** 2)# + 0.001*l2_loss
        self.loss = tf.reduce_mean((output - self.data) ** 2) + 0.001*l2_loss
        #output_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='w5')
        #self.train_op = tf.train.AdamOptimizer(0.01).minimize(self.loss, var_list=output_vars)
        self.train_op = tf.train.AdamOptimizer(0.01).minimize(self.loss)
        
x_train, y_train = mnist.train.next_batch(100) 
Network = NN()
feed = {Network.data: x_train, Network.label: y_train}
saver = tf.train.Saver({'w1':Network.w1})
#saver = tf.train.Saver({'w1':Network.w1, 'w2':Network.w2})

save_path = 'F:\python_project\model'
saver2 = tf.train.Saver({'w1':Network.w1, 'w2':Network.w2})
save_path2 = 'F:\python_project\model\tmp'
n_test_digits = 2
X_test = mnist.test.images[:2]
init = tf.global_variables_initializer()
with tf.Session() as sess:
    #if os.path.exists(save_path1):
    #    saver2.restore(sess, save_path)
    sess.run(init)
    #w1 = sess.run(Network.w1)
    #print(w1)
    for i in range(100):
        sess.run(Network.train_op, feed_dict=feed)
        if i%10 == 0:
            loss = sess.run(Network.loss, feed_dict=feed)
            print(loss)
    result = sess.run(Network.output, feed_dict={Network.data:X_test})
    #w1 = sess.run(Network.w1)
    #print(w1)
    #saver2.save(sess, save_path2)
import matplotlib.pyplot as plt

def plot_image(image, shape=[28,28]):
    plt.imshow(image.reshape(shape), cmap='Greys', interpolation='nearest')
    plt.axis('off')
    
for digit_index in range(2):
    plt.subplot(2, 2, digit_index * 2  + 1)
    plot_image(X_test[digit_index])
    plt.subplot(2, 2, digit_index * 2 + 2)
    plot_image(result[digit_index])
