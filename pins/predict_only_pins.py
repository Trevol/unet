import numpy as np
from data import testGenerator
from model import unet
import skimage.io
import skimage.transform as trans
import os


def readImage(fileName, target_size=(512, 512)):
    img = skimage.io.imread(fileName)
    img = img / 255
    img = trans.resize(img, target_size)
    # img = np.reshape(img, img.shape + (1,)) if (not flag_multi_class) else img
    img = np.reshape(img, (1,) + img.shape)
    return img


def yieldImages(dir, target_size=(512, 512), flag_multi_class=False):
    for fileName in os.listdir(dir):
        if not fileName.endswith('.jpg'):
            continue
        img = readImage(os.path.join(dir, fileName), target_size)
        yield img, fileName


def main():
    model = unet(input_size=(512, 512, 3))
    model.load_weights("unet_pins.hdf5")

    framesDir = '/home/trevol/HDD_DATA/Computer_Vision_Task/Computer_Vision_Task/frames_6'
    savePath = '/home/trevol/HDD_DATA/Computer_Vision_Task/Computer_Vision_Task/frames_6_unet_pins_only'
    for image, fileName in yieldImages(framesDir):
        results = model.predict(image, verbose=0)
        result = (results[0, :, :, 0] * 255).astype(np.uint8)
        skimage.io.imsave(os.path.join(savePath, fileName.replace('.jpg', '.png')), result)
    # saveResult("data/membrane/test", results)


def main():
    model = unet(input_size=(512, 512, 3))
    model.load_weights("unet_pins.hdf5")
    image = readImage('data/image/f_0350_23333.33_23.33.jpg')
    results = model.predict(image, batch_size=1, verbose=0)
    results


main()
