import os

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plot
from scipy.ndimage.filters import gaussian_filter1d
from scipy.signal import _savitzky_golay
from scipy.signal.filter_design import butter

image_path = 'iamDB/data/forms'


def image_segmentation(img_path):
    image = cv.imread(img_path, 0)
    image1 = image[:]
    # size = np.size(image1)
    # skel = np.zeros(image1.shape, np.uint8)
    cv.threshold(image, 170, 200, cv.THRESH_BINARY, image1)
    cv.normalize(image1, image1, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8UC1)
    data = ((np.dot(image1, np.ones(image.shape[1])) / image.shape[1]))
    # image = np.array([image[:, 0].T * data for i in range(image.shape[0])]).T

    # image1, ctrs, hierarcy = cv.findContours(image1, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # element = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
    # done = False

    # while not done:
    #     eroded = cv.erode(image1, element)
    #     temp = cv.dilate(eroded, element)
    #     temp = cv.subtract(image1, temp)
    #     skel = cv.bitwise_or(skel, temp)
    #     image1 = eroded.copy()
    #
    #     zeros = size - cv.countNonZero(image1)
    #     if zeros == size:
    #         done = True

    # Get rectangles contains each contour
    # rects = [cv.boundingRect(ctr) for ctr in ctrs]
    # print(rects)

    fig, plt = plot.subplots(1, 1)
    plt.set_ylim(-1, 2)
    plt.set_xlim(0, image.shape[1])
    plt.plot(range(0, data.size), data)
    plot.show()
    plot.imshow(image1, aspect='auto', cmap='Greys_r')
    plot.show()


if __name__ == '__main__':
    for pth in os.listdir(image_path):
        image_segmentation(os.path.join(image_path, pth))
