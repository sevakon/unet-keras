from keras.models import Model
from keras.optimizers import *
from model.utils import input_tensor, single_conv, double_conv, deconv, pooling, merge, callback

class UNet(Model):
    """ U-Net atchitecture
    Creating a U-Net class that inherits from keras.models.Model
    In initializer, CNN layers are defined using functions from model.utils
    Then parent-initializer is called wuth calculated input and output layers
    Build function is also defined for model compilation and summary
    checkpoint returns a ModelCheckpoint for best model fitting
    """
    def __init__(
        self,
        input_size,
        n_filters,
        pretrained_weights = None
    ):
        # define input layer
        input = input_tensor(input_size)

        # begin with contraction part
        conv1 = double_conv(input, n_filters * 1)
        pool1 = pooling(conv1)

        conv2 = double_conv(pool1, n_filters * 2)
        pool2 = pooling(conv2)

        conv3 = double_conv(pool2, n_filters * 4)
        pool3 = pooling(conv3)

        conv4 = double_conv(pool3, n_filters * 8)
        pool4 = pooling(conv4)

        conv5 = double_conv(pool4, n_filters * 16)

        # expansive path
        up6 = deconv(conv5, n_filters * 8)
        up6 = merge(conv4, up6)
        conv6 = double_conv(up6, n_filters * 8)

        up7 = deconv(conv6, n_filters * 4)
        up7 = merge(conv3, up7)
        conv7 = double_conv(up7, n_filters * 4)

        up8 = deconv(conv7, n_filters * 2)
        up8 = merge(conv2, up8)
        conv8 = double_conv(up8, n_filters * 2)

        up9 = deconv(conv8, n_filters * 1)
        up9 = merge(conv1, up9)
        conv9 = double_conv(up9, n_filters * 1)

        # define output layer
        output = single_conv(conv9, 1, 1)

        # initialize Keras Model with defined above input and output layers
        super(UNet, self).__init__(inputs = input, outputs = output)
        
        # load preatrained weights
        if pretrained_weights:
            self.load_weights(pretrained_weights)

    def build(self):
        self.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])
        self.summary()

    def save_model(self, name):
        self.save_weights(name)

    @staticmethod
    def checkpoint(name):
        return callback(name)
    
# TODO: FIX SAVING MODEL: AT THIS POINT, ONLY SAVING MODEL WEIGHTS IS AVAILBILE
# SINCE SUBSCLASSING FROM KERAS.MODEL RESTRICTS SAVING MODEL AS AN HDF5 FILE
