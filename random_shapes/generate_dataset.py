import os

import numpy as np
import cv2


def generate_shapes(img_shape=(512, 512)):
    bgColor = 120
    shapeColor = 220
    minSide = 40
    maxSide = 130
    img = np.full(img_shape, 120, np.uint8)
    w, h = np.random.randint(minSide, maxSide, 2, np.int32)
    imgH, imgW = img_shape
    y, = np.random.randint(1, imgH - h, 1, int)
    x, = np.random.randint(1, imgW - w, 1, int)
    img[y:y + h, x:x + w] = shapeColor
    mask = np.empty_like(img)
    mask[img == bgColor] = 0
    mask[img == shapeColor] = 255
    return img, mask


def main():
    os.makedirs('data/image', exist_ok=True)
    os.makedirs('data/label', exist_ok=True)
    for i in range(30):
        img, mask = generate_shapes()
        cv2.imwrite(f'data/image/{i}.png', img)
        cv2.imwrite(f'data/label/{i}.png', mask)
        # cv2.imshow('img', img)
        # cv2.imshow('mask', mask)
        # if cv2.waitKey() == 27:
        #     return



main()
