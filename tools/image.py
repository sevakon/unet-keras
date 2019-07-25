import glob
import skimage.io as io
import skimage.transform as trans
import numpy as np
import pylab as plt


def square_image(img, random = None):
    """ Square Image
    Function that takes an image (ndarray),
    gets its maximum dimension,
    creates a black square canvas of max dimension
    and puts the original image into the
    black canvas's center
    If random [0, 2] is specified, the original image is placed
    in the new image depending on the coefficient,
    where 0 - constrained to the left/up anchor,
    2 - constrained to the right/bottom anchor
    """
    size = max(img.shape[0], img.shape[1])
    new_img = np.zeros((size, size),np.float32)
    ax, ay = (size - img.shape[1])//2, (size - img.shape[0])//2

    if random and not ax == 0:
        ax = int(ax * random)
    elif random and not ay == 0:
        ay = int(ay * random)

    new_img[ay:img.shape[0] + ay, ax:ax+img.shape[1]] = img
    return new_img


def reshape_image(img, target_size):
    """ Reshape Image
    Function that takes an image
    and rescales it to target_size
    """
    img = trans.resize(img,target_size)
    img = np.reshape(img,img.shape+(1,))
    img = np.reshape(img,(1,)+img.shape)
    return img

def normalize_mask(mask):
    """ Mask Normalization
    Function that returns normalized mask
    Each pixel is either 0 or 1
    """
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0
    return mask

def show_image(img):
    plt.imshow(img, cmap=plt.cm.gray)
    plt.show()
