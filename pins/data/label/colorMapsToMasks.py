import cv2
import numpy as np

def main():
    im = cv2.imread("f_0669_44600.00_44.60.png")
    print(np.unique(im[..., 0]))
    print(np.unique(im[..., 1]))
    print(np.unique(im[..., 2]))


main()
