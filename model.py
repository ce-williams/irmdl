# run using python not python3
# tensorflow library import
import tensorflow as tf

# import matplot lib
import matplotlib.pyplot as plt


# images of handwritten digits 0-9 28x28 px
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# normalize data to reduce data values between 0 and 1
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)


# 
model = tf.keras.models.Sequential()
# input layer, multidimensional needs to be flattened to simplify network
# filter out large unneeded information
model.add(tf.keras.layers.Flatten())
# default activation function
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
# another layer added 
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
# output layer, output will need to be the number of classifications
# Because we are trying to guess a number 0-9, our output is 10
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

# Params for training the model
model.compile(
    # Optimizer: most complex part of network.
    # depending on attributes of input, other opts avail
    # adam optimizer: default optimizer
            optimizer='adam',
            # many ways to calc loss, scc default
            loss='sparse_categorical_crossentropy',
            # metrics looking to track 
            metrics=['accuracy']
            )
# important to not 'overfit' model.
# intended outcome is to train to understand attrs
# of classified input, rather than just memorize.
model.fit(x_train, y_train, epochs=3)

# calculations to determine confidence of accuracy, and loss
val_loss, val_acc = model.evaluate(x_test, y_test)

print(val_loss, val_acc)




# graph of number, reduced color map to to B&W
plt.imshow(x_train[0], cmap = plt.cm.binary)
plt.show()

print(x_train[0])