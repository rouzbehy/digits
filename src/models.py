import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization, Dense, Flatten
from tensorflow.keras.models import Model

class ConvModel(Model):

    def __init__(self, shape):
        super().__init__()
        
        self.first_conv = Conv2D(filters=32, 
                                 kernel_size=(4,4),
                                 input_shape=shape,
                                 name='Conv2D-1',
                                 kernel_initializer='he_uniform')
        
        self.max_pool_1 = MaxPooling2D(pool_size=(3,3))

        self.second_conv = Conv2D(filters=64,
                                  kernel_size=(4,4),
                                  name='Conv2D-2',
                                  kernel_initializer='he_uniform')
        self.normalization = BatchNormalization()
        #self.third_conv = Conv2D(filters=64,
        #                          kernel_size=(3,3),
        #                          name='Conv2D-3')
        
        #self.max_pool_2 = MaxPooling2D(pool_size=(2,2))

        self.flattening = Flatten()

        self.dense_layer = Dense(256, activation='relu',
                                 kernel_initializer='he_uniform')

        self.decision = Dense(10, activation='softmax',
                              kernel_initializer='he_uniform')
    
    def call(self, x):
        tmp = self.first_conv(x)
        #print(f"after first conv: {tmp.shape}")
        tmp = self.max_pool_1(tmp)
        #print(f"after first max pool: {tmp.shape}")
        tmp = self.second_conv(tmp)
        #
        tmp = self.normalization(tmp)
        #print(f"after second conv: {tmp.shape}")
        #tmp = self.third_conv(tmp)
        #print(f"after third conv: {tmp.shape}")
        #tmp = self.max_pool_2(tmp)
        #print(f"after second max pool: {tmp.shape}")
        tmp = self.flattening(tmp)
        #print(f"after flattening: {tmp.shape}")
        tmp = self.dense_layer(tmp)
        #print(f"after dense: {tmp.shape}")
        tmp =  self.decision(tmp)
        #print(f"after decision dense: {tmp.shape}")
        return tmp