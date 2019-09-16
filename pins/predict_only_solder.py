from time import time

import cv2
import numpy as np
from data import testGenerator
from model import unet
import skimage.io
import skimage.transform as trans
import os


def readImage(fileName, target_size=(512, 512), as_gray=False):
    img = skimage.io.imread(fileName, as_gray=as_gray)
    img = originalImage = trans.resize(img, target_size)

    if img.dtype == np.uint8:
        img = img / 255
    img = np.reshape(img, img.shape + (1,)) if as_gray else img
    img = np.reshape(img, (1,) + img.shape)
    return img, originalImage


def yieldImages(dir, target_size=(512, 512), as_gray=False):
    for fileName in sorted(os.listdir(dir)):
        if not fileName.endswith('.jpg'):
            continue
        img, originalImage = readImage(os.path.join(dir, fileName), target_size, as_gray)
        yield img, originalImage, fileName


def playResults():
    framesDir = '/home/trevol/HDD_DATA/Computer_Vision_Task/Computer_Vision_Task/frames_6'
    savePath = '/home/trevol/HDD_DATA/Computer_Vision_Task/Computer_Vision_Task/frames_6_unet_solder_only'
    for fileName in sorted(os.listdir(savePath)):
        if not fileName.endswith('.png'):
            continue
        resultPath = os.path.join(savePath, fileName)
        framePath = os.path.join(framesDir, fileName.replace('.png', '.jpg'))
        result = cv2.imread(resultPath)
        original = cv2.imread(framePath)
        cv2.imshow('original', original)
        cv2.imshow('result', result)
        if cv2.waitKey(1) == 27:
            break


def predict256x256x3_n_save():
    targetSize = (256, 256)
    model = unet(input_size=targetSize + (3,))
    model.load_weights("checkpoints/rgb/unet_pins_20_0.001_1.000.hdf5")

    framesDir = '/home/trevol/HDD_DATA/Computer_Vision_Task/Computer_Vision_Task/frames_6'
    savePath = '/home/trevol/HDD_DATA/Computer_Vision_Task/Computer_Vision_Task/frames_6_unet_pins_only'
    for batch, originalImage, fileName in yieldImages(framesDir, targetSize, as_gray=False):
        results = model.predict(batch, verbose=0)
        results = np.round((results[0, :, :, 0] * 255), 0).astype(np.uint8)
        skimage.io.imsave(os.path.join(savePath, fileName.replace('.jpg', '.png')), results)


def predict512x512x3_n_save():
    targetSize = (512, 512)
    model = unet(input_size=targetSize + (3,))
    model.load_weights("checkpoints/rgb/unet_pins_20_0.001_1.000.hdf5")

    framesDir = '/home/trevol/HDD_DATA/Computer_Vision_Task/Computer_Vision_Task/frames_6'
    savePath = '/home/trevol/HDD_DATA/Computer_Vision_Task/Computer_Vision_Task/frames_6_unet_solder_only'
    os.makedirs(savePath, exist_ok=True)
    for batch, originalImage, fileName in yieldImages(framesDir, targetSize, as_gray=False):
        results = model.predict(batch, verbose=0)
        results = np.round((results[0, :, :, 0] * 255), 0).astype(np.uint8)
        skimage.io.imsave(os.path.join(savePath, fileName.replace('.jpg', '.png')), results)


def predict512x512x3_n_show():
    targetSize = (512, 512)
    model = unet(input_size=targetSize + (3,))
    model.load_weights("checkpoints/rgb_solder/unet_solder512x512_2_0.003_0.999.hdf5")

    framesDir = '/home/trevol/HDD_DATA/Computer_Vision_Task/Computer_Vision_Task/frames_6'
    for batch, originalImage, fileName in yieldImages(framesDir, targetSize, as_gray=False):
        t0 = time()
        results = model.predict(batch, verbose=0)
        t1 = time()
        print(t1 - t0)
        results = np.round((results[0, :, :, 0] * 255), 0).astype(np.uint8)
        cv2.imshow('image', originalImage)
        cv2.imshow('results', results)
        if cv2.waitKey(1) == 27:
            break


def predict256x256x3_n_show():
    targetSize = (256, 256)
    model = unet(input_size=targetSize + (3,))
    model.load_weights("checkpoints/rgb_solder/unet_solder512x512_2_0.003_0.999.hdf5")

    framesDir = '/home/trevol/HDD_DATA/Computer_Vision_Task/Computer_Vision_Task/frames_6'
    for batch, originalImage, fileName in yieldImages(framesDir, targetSize, as_gray=False):
        t0 = time()
        results = model.predict(batch, verbose=0)
        t1 = time()
        print(t1 - t0)
        results = np.round((results[0, :, :, 0] * 255), 0).astype(np.uint8)
        cv2.imshow('image', originalImage)
        cv2.imshow('results', results)
        if cv2.waitKey(1) == 27:
            break


def main():
    playResults()


main()
