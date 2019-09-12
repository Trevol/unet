import cv2

from data import trainGenerator
import numpy as np


def main():
    data_gen_args = dict(rotation_range=0.2,
                         width_shift_range=0.05,
                         height_shift_range=0.05,
                         shear_range=0.05,
                         zoom_range=0.05,
                         horizontal_flip=True,
                         fill_mode='nearest')
    data_gen_args = dict()

    colorMode = "grayscale"
    myGene = trainGenerator(1, 'data', 'image', 'pin_only_masks', data_gen_args, image_color_mode=colorMode,
                            target_size=(1080 // 2, 1920 // 2),
                            save_to_dir=None)
    for _ in range(20):
        img, mask = next(myGene)
        img = np.squeeze(img[0])
        mask = np.squeeze(mask[0])
        img = np.uint8(img * 255)
        mask = np.uint8(mask * 255)
        cv2.imshow('img', img)
        cv2.imshow('mask', mask)
        cv2.waitKey()


main()
