import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.layers import BatchNormalization, Dense, Flatten
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
class ConvModel(Model):

    def __init__(self, shape, filters):
        super().__init__()
        
        self.first_conv = Conv2D(filters=filters, 
                                 kernel_size=(4,4),
                                 input_shape=shape,
                                 name='Conv2D-1',
                                 kernel_initializer='he_uniform')
        
        self.max_pool_1 = MaxPooling2D(pool_size=(3,3))

        self.normalization = BatchNormalization()

        self.second_conv = Conv2D(filters=filters/2,
                                  kernel_size=(4,4),
                                  name='Conv2D-2',
                                  kernel_initializer='he_uniform')
        
        self.flattening = Flatten()

        self.dense_layer = Dense(1024, activation='relu',
                                kernel_initializer='he_uniform')
        
        self.drop_out = Dropout(0.2)

        self.decision = Dense(10, activation='softmax',
                              kernel_initializer='he_uniform')
    
    def call(self, x):
        #print("x.shape:", x.shape)
        tmp = self.first_conv(x)
        #print("tmp.shape: ", tmp.shape)
        tmp = self.max_pool_1(tmp)
        #print("tmp.shape: ", tmp.shape)
        tmp = self.second_conv(tmp)
        #print("tmp.shape: ", tmp.shape)
        tmp = self.normalization(tmp)
        #print("tmp.shape: ", tmp.shape)
        tmp = self.flattening(tmp)
        #print("tmp.shape: ", tmp.shape)
        tmp = self.dense_layer(tmp)
        #print("tmp.shape: ", tmp.shape)
        tmp = self.drop_out(tmp)
        tmp = self.decision(tmp)
        #print("tmp.shape: ", tmp.shape)
        return tmp

def plot_utility(history, target='loss', axis=None):
    ax = ''
    if axis:
        ax = axis
    else:
        _, axis = plt.subplots()
        ax = axis
    ax.plot(history.history[target], color='blue')
    ax.plot(history.history['val_'+target], color='green')
    return ax