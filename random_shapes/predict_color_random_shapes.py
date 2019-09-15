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
    model = unet(input_size=targetSize + (3,))
    model.load_weights("checkpoints/color_shapes/unet_rgb_shapes_10_0.1244_0.975.hdf5")

    while True:
        shapeImage, _ = generate_shapes(img_shape=(512, 512, 3), bgColor=[117, 122, 125], shapeColor=[180, 211, 250])
        batch = adjustImage(shapeImage, targetSize, asGray=False)
        results = model.predict(batch, batch_size=1, verbose=0)

        results = np.round(np.squeeze(results[0]) * 255, 0).astype(np.uint8)

        cv2.imshow('image', shapeImage)
        cv2.imshow('results', results)

        while True:
            key = cv2.waitKey()
            if key == 27:
                return
            if key == ord('d'):
                break
        cv2.imshow('image', np.zeros_like(shapeImage))
        cv2.imshow('results', np.zeros_like(shapeImage))


main()
