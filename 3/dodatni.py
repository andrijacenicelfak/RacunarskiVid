import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def morphological_reconstruction(marker: np.ndarray, mask: np.ndarray):
    kernel = np.ones(shape=(7, 7), dtype=np.uint8) * 255
    while True:
        expanded = cv.dilate(src=marker, kernel=kernel)
        expanded = cv.bitwise_and(src1=expanded, src2=mask)

        if (marker == expanded).all():
            return expanded
        marker = expanded

if __name__ == '__main__':
    img = cv.cvtColor(cv.imread("slika2.jpg"), cv.COLOR_BGR2GRAY)
    _, bin_img = cv.threshold(img, 245, 1, cv.THRESH_BINARY)

    width = bin_img.shape[0]
    height = bin_img.shape[1]

    with_box = cv.rectangle(np.zeros(bin_img.shape, np.uint8),(0,0), (height-1, width-1), 1, 1)

    px_edge = cv.bitwise_and(src1 = with_box,src2 = bin_img)

    reconstructed = morphological_reconstruction(px_edge, bin_img)

    no_edge = cv.bitwise_xor(bin_img, reconstructed)
    plt.imshow(bin_img)
    plt.show()