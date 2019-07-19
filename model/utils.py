from keras.layers import Input, Conv2D, Conv2DTranspose, BatchNormalization, Activation, MaxPooling2D, Dropout, concatenate
from keras.callbacks import ModelCheckpoint

''' utils.py
File that defines layers for u-net model.
1. input_tensor - Input layer
2. single_conv - one 2D Convolutional layer
3. double_conv - two Sequential 2D Convolutional layers
4. deconv - one 2D Transposed Convolutional layer
5, pooling - one Max Pooling layer followed by Dropout function
6. merge - concatenates two layers
7. callback - returns a ModelCheckpoint, used in main.py for model fitting
'''

# function that defines input layers for given shape
def input_tensor(input_size):
    x = Input(input_size)
    return x

# function that defines one convolutional layer with certain number of filters
def single_conv(input_tensor, n_filters, kernel_size):
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size), activation = 'sigmoid')(input_tensor)
    return x

# function that defines two sequential 2D convolutional layers with certain number of filters
def double_conv(input_tensor, n_filters, kernel_size = 3):
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size), padding = 'same', kernel_initializer = 'he_normal')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size), padding = 'same', kernel_initializer = 'he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

# function that defines 2D transposed convolutional (Deconvolutional) layer
def deconv(input_tensor, n_filters, kernel_size = 3, stride = 2):
    x = Conv2DTranspose(filters = n_filters, kernel_size = (kernel_size, kernel_size), strides = (stride, stride), padding = 'same')(input_tensor)
    return x

# function that defines Max Pooling layer with pool size 2 and applies Dropout
def pooling(input_tensor, dropout_rate = 0.1):
    x = MaxPooling2D(pool_size = (2, 2))(input_tensor)
    x = Dropout(rate = dropout_rate)(x)
    return x

# function that merges two layers (Concatenate)
def merge(input1, input2):
    x = concatenate([input1, input2])
    return x

# function to create ModelCheckpoint
def callback(name):
    return ModelCheckpoint(name, monitor='loss',verbose=1, save_best_only=True)
