import tensorflow as tf
from tensorflow import keras 
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
import models
## Hide GPU from visible devices
## tf is much faster on CPUS when running on an M1 Mac
tf.config.set_visible_devices([], 'GPU')

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
# reshape dataset to have a single channel
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))
# one hot encode target values
y_train = keras.utils.to_categorical(y_train)
y_test  = keras.utils.to_categorical(y_test)

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.


print(f"shape of x_train: {x_train.shape}")
print(f"shape of x_test: {x_test.shape}")

shape = (None,28,28,1)

model = models.ConvModel(shape)
model.compile(optimizer= 'adam', 
              loss     = 'categorical_crossentropy',
              metrics  = ['accuracy'])

model.fit(x_train, y_train,
                epochs=1,
                shuffle=True,
                validation_data=(x_test, y_test),
                batch_size=32)

print(model.summary())