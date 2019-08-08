import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
data_dir = 'F:\python_project\mnist'

mnist = input_data.read_data_sets(data_dir, one_hot=True)

class NN():
    def __init__(self):
        self.x = tf.placeholder(shape=[None, 784], dtype=tf.float32)
        self.model()
    def model(self):
        w1 = tf.Variable(tf.truncated_normal(shape=[784, 500], dtype=tf.float32, stddev=0.1))
        w2 = tf.Variable(tf.truncated_normal(shape=[500, 150], dtype=tf.float32, stddev=0.1))
        w3 = tf.Variable(tf.truncated_normal(shape=[150, 500], dtype=tf.float32, stddev=0.1))
        w4 = tf.Variable(tf.truncated_normal(shape=[500, 784], dtype=tf.float32, stddev=0.1))
        b1 = tf.Variable(tf.zeros(shape=[500], dtype=tf.float32))
        b2 = tf.Variable(tf.zeros(shape=[150], dtype=tf.float32))
        b3 = tf.Variable(tf.zeros(shape=[500], dtype=tf.float32))
        b4 = tf.Variable(tf.zeros(shape=[784], dtype=tf.float32))
        
        h1 = tf.nn.relu(tf.add(tf.matmul(self.x, w1), b1))
        h2 = tf.nn.relu(tf.add(tf.matmul(h1, w2), b2))
        h3 = tf.nn.relu(tf.add(tf.matmul(h2, w3), b3))
        self.h4 = h4 = (tf.add(tf.matmul(h3, w4), b4))
        
        with tf.name_scope('step1') as scope:
            h5 = (tf.add(tf.matmul(h1, w4), b4))
            l2_loss1 = tf.nn.l2_loss(w1)
            self.cost1 = cost1 = tf.reduce_mean((self.x - h5) ** 2) + 0.001*l2_loss1
            self.train_op1 = tf.train.AdamOptimizer(0.01).minimize(cost1)
        
        with tf.name_scope('step2') as scope:
            h6 = tf.add(tf.matmul(h2, w3), b3)
            l2_loss2 = tf.nn.l2_loss(w2)
            self.cost2 = cost2 = tf.reduce_mean((h6 - h1) ** 2) + 0.001*l2_loss2
            var_train = [w2, b2, w3, b3]
            self.train_op2 = tf.train.AdamOptimizer(0.01).minimize(cost2, var_list=var_train)
        
        with tf.name_scope('step3') as scope:
            l2_loss3 = tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2)
            self.cost3 = cost3 = tf.reduce_mean((self.x - h4) ** 2) + 0.001*l2_loss3
            self.train_op3 = tf.train.AdamOptimizer(0.01).minimize(cost3)
        
x_train, y_train = mnist.train.next_batch(100)
network = NN()
X_test = mnist.test.images[:2]
init = tf.global_variables_initializer()
feed = {network.x:x_train}
iteration1 = 100
iteration2 = 1000
iteration3 = 10
with tf.Session() as sess:
    sess.run(init)
    for i in range(iteration1):
        sess.run(network.train_op1, feed_dict=feed)
        if i%10 == 0:
            loss1 = sess.run(network.cost1, feed_dict=feed)
            print(loss1)
    for j in range(iteration2):
        sess.run(network.train_op2, feed_dict=feed)
        if j%100 == 0:
            loss2 = sess.run(network.cost2, feed_dict=feed)
            print(loss2)
    #for k in range(iteration3):
    #    sess.run(network.train_op3, feed_dict=feed)
    #    loss3 = sess.run(network.cost3, feed_dict=feed)
    #    print(loss3)
    result = sess.run(network.h4, feed_dict={network.x:X_test})
n_test_digits = 2
import matplotlib.pyplot as plt
def plot_image(image, shape=[28,28]):
    plt.imshow(image.reshape(shape), cmap='Greys', interpolation='nearest')
    plt.axis('off')
for digit_index in range(2):
    plt.subplot(2, 2, digit_index * 2 + 1)
    plot_image(X_test[digit_index])
    plt.subplot(2, 2, digit_index * 2 + 2)
    plot_image(result[digit_index])

    
