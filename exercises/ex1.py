import tensorflow as tf

session = tf.Session()

a = tf.constant(value=10)
b = tf.constant(value=32)

result = session.run(fetches=(a * b))

print(result)
