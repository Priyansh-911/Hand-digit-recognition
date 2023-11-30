# import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist as mnist
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, Activation

(x_train, y_train) ,(x_test, y_test) = mnist.load_data()


x_train = tf.keras.utils.normalize(x_train,  axis = 1)
x_test = tf.keras.utils.normalize(x_test,  axis = 1)

x_train_re = x_train.reshape(x_train.shape[0] , 28, 28, 1)
x_test_re = x_test.reshape(x_test.shape[0] , 28, 28, 1)



model = Sequential()

#Convolutional layer 1
model.add(Conv2D(64, (3, 3), input_shape = x_train_re.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

#Convolutional layer 2
model.add(Conv2D(64, (3, 3), activation=tf.keras.activations.relu))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

#Flatten layer
model.add(Flatten())

#full connected layer
model.add(Dense(64))
model.add(Activation('relu'))

model.add(Dense(32))
model.add(Activation('relu'))

model.add(Dense(10))
model.add(Activation('softmax'))
model.summary()

model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

mod = model.fit(x_train_re, y_train, epochs = 5)

test_loss, test_acc = model.evaluate(x_test_re, y_test)
print("Test Loss on 10,000 test samples", test_loss)
print("Test Accuracy on 10,000 test samples", test_acc)

predictions = model.predict([x_test_re], verbose = 0)
print(predictions)

plt.imshow(x_test[0])
print(np.argmax(predictions[0]))

model.save('digits_model_new.h5', mod)



