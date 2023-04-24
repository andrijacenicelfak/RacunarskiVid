import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


def fft(img):
    fft = np.fft.fft2(img)
    fft_shift = np.fft.fftshift(fft)
    return fft_shift


def inverse_fft(img):
    #ifft_shift = np.fft.ifftshift(img).real
    ifft = np.fft.ifft2(img)
    return np.uint8(np.abs(ifft))


def magnetude_spec(img):
    return np.log(np.abs(imgfft))/15


def createmask(img, treshold, removeradius=0):
    height = img.shape[0]
    width = img.shape[1]
    _, mask = cv.threshold(img, treshold, 1, cv.THRESH_BINARY)
    mask = np.abs(mask)
    if removeradius > 0:
        mask = cv.circle(mask, (height//2, width//2),
                         removeradius, 0, thickness=cv.FILLED)
    return 1 - mask


def createmask2(img):
    plt.imshow(img*2.8, cmap="gray")
    plt.show()
    img = img.copy()
    img *= img
    mask = cv.adaptiveThreshold(np.uint8(img*8),
                                1,
                                cv.THRESH_BINARY_INV,
                                cv.THRESH_BINARY_INV,
                                3, -0.5)
    plt.imshow(mask, cmap="gray")
    plt.show()
    return mask


if __name__ == '__main__':
    img = cv.imread("input.png")
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    imgfft = fft(img)
    mag_spec = magnetude_spec(imgfft)

    #mask = createmask2(mag_spec)
    mask = createmask(mag_spec, 0.9, 8)

    imgfiltered = imgfft * mask
    magnetude_spec_mod = magnetude_spec(imgfiltered)
    plt.imshow(mask, cmap="gray")
    plt.show()

    img2filtered = inverse_fft(imgfiltered)

    cv.imwrite("fft_mag.png", mag_spec*255)
    cv.imwrite("fft_mag_filtered.png", magnetude_spec_mod*255)
    cv.imwrite("output.png", img2filtered)
    cv.imshow("Mask", mask)
    cv.imshow("Original", img)
    cv.imshow("Magnitude spec", mag_spec)
    cv.imshow("Magnitude spec mod", magnetude_spec_mod)
    cv.imshow("Filtered", img2filtered)
    cv.waitKey(0)
    cv.destroyAllWindows()
