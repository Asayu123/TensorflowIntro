from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# Setup Mnist Data
mnist = input_data.read_data_sets(train_dir="MNIST_data/", one_hot=True)

# Setup Placeholder
x = tf.placeholder(tf.float32, [None, 784])  # Serialized Image Pixel Data
y_ = tf.placeholder(tf.float32, [None, 10])  # Correct answers
W = tf.Variable(tf.zeros([784, 10]))  # Initial Weight
b = tf.Variable(tf.zeros([10]))  # Initial Bias

# Define output function
y = tf.nn.softmax(tf.matmul(x, W) + b)
# Define evaluate function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
# cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

# Define Training parameters
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# Execute Operations
with tf.Session() as session:
    # Initialize Data
    init = tf.global_variables_initializer()
    session.run(init)

    # Start Learning
    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        session.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    # Define prediction and accuracy
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Calculate Accuracy and print result.
    print(session.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

