# tensorflow library import
import tensorflow as tf


# images of handwritten digits 0-9 28x28 px
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()


# import matplot lib
import matplotlib.pyplot as plt

plt.imshow(x_train[0])
plt.show()