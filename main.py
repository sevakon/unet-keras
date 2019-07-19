from model.unet import UNet
from tools.data import train_generator, test_generator, save_results

# TODO: move to config .json files
img_height = 512
img_width = 512
img_size = (img_height, img_width)
train_path = '/Users/vsevolod.konyahin/Desktop/DataSet/train'
test_path = '/Users/vsevolod.konyahin/Desktop/DataSet/test'
save_path = '/Users/vsevolod.konyahin/Desktop/DataSet/results'
model_name = 'unet_model.hdf5'
model_weights_name = 'unet_weight_model.hdf5'

if __name__ == "__main__":
    # generates training set
    train_gen = train_generator(2, train_path, 'img', 'mask', img_size)

    # build model
    unet = UNet(input_size = (img_width,img_height,1), n_filters = 64)
    unet.build()

    # createting a callback, hence best weights configurations will be saved
    model_checkpoint = unet.checkpoint(model_name)

    # model training
    unet.fit_generator(train_gen, steps_per_epoch = 300, epochs = 1, callbacks=[model_checkpoint])

    # generated testing set
    test_gen = test_generator(test_path, 30, img_size)

    # display results
    results = unet.predict_generator(test_gen,30,verbose=1)
    save_results(save_path, results)

    # saving model weights
    unet.save_model(model_weights_name)
