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
    org_image = cv.imread("coins.png")
    grayscale_image = cv.cvtColor(org_image, cv.COLOR_BGR2GRAY)

    _, coins = cv.threshold(
        grayscale_image, 190, 255, cv.THRESH_BINARY_INV)

    coins = cv.morphologyEx(
        coins, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5)))

    coins = cv.morphologyEx(
        coins, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_ELLIPSE, (15, 15)))

    hsv = cv.cvtColor(org_image, cv.COLOR_BGR2HSV)
    hue = hsv[:, :, 1]

    _, mask = cv.threshold(hue, 50, 255, cv.THRESH_BINARY)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN,
                           cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5)))

    reconstructed = morphological_reconstruction(mask, coins)

    cv.imwrite("coin_mask.png", reconstructed)

    cv.imshow("original", org_image)
    cv.imshow("mask", reconstructed)
    cv.imshow("coins", coins)

    cv.waitKey(0)
    cv.destroyAllWindows()
