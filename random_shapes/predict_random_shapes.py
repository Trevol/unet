import cv2
import numpy as np
from data import testGenerator
from model import unet
import skimage.io
import skimage.transform as trans
import os

from random_shapes.generate_dataset import generate_shapes


def readImage(fileName, target_size=(512, 512), as_gray=False):
    img = skimage.io.imread(fileName, as_gray=as_gray)
    return adjustImage(img, target_size, as_gray)


def adjustImage(img, targetSize, asGray=False):
    if img.dtype == np.uint8:
        img = img / 255
    img = trans.resize(img, targetSize)
    img = np.reshape(img, img.shape + (1,)) if asGray else img
    img = np.reshape(img, (1,) + img.shape)
    return img


def main():
    targetSize = (512, 512)
    model = unet(input_size=targetSize + (1,))
    model.load_weights("checkpoints/unet_grayscale_shapes_2_0.0000_1.000_.hdf5")

    while True:
        shapeImage, _ = generate_shapes(targetSize)
        batch = adjustImage(shapeImage, targetSize, True)
        results = model.predict(batch, batch_size=1, verbose=0)

        results = np.round(np.squeeze(results[0]) * 255, 0).astype(np.uint8)

        cv2.imshow('image', shapeImage)
        cv2.imshow('results', results)
        if cv2.waitKey() == 27:
            return
        cv2.imshow('image', np.zeros_like(shapeImage))
        cv2.imshow('results', np.zeros_like(shapeImage))


# def main_():
#     model = unet()
#     model.load_weights("../unet_membrane_5_0.123_0.946.hdf5")
#     # image = readImage('../data/membrane/test/0.png', target_size=(256, 256), as_gray=True)
#     image = readImage('data/image/f_0350_23333.33_23.33.jpg', target_size=(256, 256), as_gray=True)
#     results = model.predict(image, batch_size=1, verbose=0)
#
#     image = np.round(np.squeeze(image[0])*255, 0).astype(np.uint8)
#     results = np.round(np.squeeze(results[0])*255, 0).astype(np.uint8)
#
#     cv2.imshow('image', image)
#     cv2.imshow('results', results)
#     cv2.waitKey()


main()
