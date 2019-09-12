import cv2
import numpy as np
from data import testGenerator
from model import unet
import skimage.io
import skimage.transform as trans
import os


def readImage(fileName, target_size=(512, 512), as_gray=False):
    img = skimage.io.imread(fileName, as_gray=as_gray)
    img = img / 255
    img = trans.resize(img, target_size)
    img = np.reshape(img, img.shape + (1,)) if as_gray else img
    img = np.reshape(img, (1,) + img.shape)
    return img


def yieldImages(dir, target_size=(512, 512), flag_multi_class=False):
    for fileName in sorted(os.listdir(dir)):
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
    model = unet(input_size=(512, 512, 1))
    model.load_weights("checkpoints/unet_grayscale_pins_1_0.6830_0.949.hdf5")

    image = readImage('data/image/f_0350_23333.33_23.33.jpg', as_gray=True)
    results = model.predict(image, batch_size=1, verbose=0)

    image = np.round(np.squeeze(image[0]) * 255, 0).astype(np.uint8)
    results = np.round(np.squeeze(results[0]) * 255, 0).astype(np.uint8)

    cv2.imshow('image', image)
    cv2.imshow('results', results)
    cv2.waitKey()


def main_():
    model = unet()
    model.load_weights("../unet_membrane.hdf5")
    image = readImage('../data/membrane/test/0.png', target_size=(256, 256), as_gray=True)
    results = model.predict(image, batch_size=1, verbose=0)

    image = np.round(np.squeeze(image[0])*255, 0).astype(np.uint8)
    results = np.round(np.squeeze(results[0])*255, 0).astype(np.uint8)

    cv2.imshow('image', image)
    cv2.imshow('results', results)
    cv2.waitKey()


main()
