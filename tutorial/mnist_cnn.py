from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(value=0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(value=x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# Main Logic Starts Here
with tf.Session() as sess:
    # data setup
    mnist = input_data.read_data_sets(train_dir="MNIST_data/", one_hot=True)

    # placeholders and variables setup
    x = tf.placeholder(tf.float32, [None, 784])  # Serialized Image Pixel Data
    y_ = tf.placeholder(tf.float32, [None, 10])  # Correct answers

    x_image = tf.reshape(tensor=x, shape=[-1, 28, 28, 1])

    # Define Convolution Layer
    W_conv1 = weight_variable(shape=[5, 5, 1, 32])
    W_conv2 = weight_variable(shape=[5, 5, 32, 64])

    b_conv1 = bias_variable(shape=[32])
    b_conv2 = bias_variable(shape=[64])

    # Define Convolution Operation and Pools
    h_conv1 = tf.nn.relu(features=conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    h_conv2 = tf.nn.relu(features=conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # Join Operation
    W_fc1 = weight_variable(shape=[7 * 7 * 64, 1024])
    b_fc1 = bias_variable(shape=[1024])
    h_pool2_flat = tf.reshape(tensor=h_pool2, shape=[-1, 7*7*64])
    h_fc1 = tf.nn.relu(features=tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # parameter for dropout
    keep_prob = tf.placeholder(dtype=tf.float32)
    h_fc1_drop = tf.nn.dropout(x=h_fc1, keep_prob=keep_prob)
    W_fc2 = weight_variable(shape=[1024, 10])
    b_fc2 = bias_variable(shape=[10])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    # Evaluator definition
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(input_tensor=tf.cast(x=correct_prediction, dtype=tf.float32))

    # Start Training
    sess.run(tf.global_variables_initializer())
    for i in range(20000):
        batch = mnist.train.next_batch(50)
        if i % 100 == 0:
            current_train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            print("Loop:{0}, Accuracy:{1}".format(i, current_train_accuracy))

        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    print("test accuracy %g" % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))








