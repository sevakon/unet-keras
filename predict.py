from model.unet import UNet
from tools.data import test_generator, save_results
import sys

img_height = 512
img_width = 512
img_size = (img_height, img_width)
test_path = 'data/test'
save_path = 'data/results'
model_weights_name = 'unet_bones_weights.hdf5'

if __name__ == "__main__":
    """ Prediction Script
    Run this Python script with a command line
    argument that defines number of test samples
    e.g. python predict.py 6
    Note that test samples names should be:
    1.jpg, 2.jpg, 3.jpg ...
    """

    # get number of samples from command line
    samples_number = int(sys.argv[1])

    # build model
    unet = UNet(
        input_size = (img_width,img_height,1),
        n_filters = 64,
        pretrained_weights = model_weights_name
    )
    unet.build()

    # generated testing set
    test_gen = test_generator(test_path, samples_number, img_size)

    # display results
    results = unet.predict_generator(test_gen, samples_number ,verbose=1)
    save_results(save_path, results)
