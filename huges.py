import numpy as np
import cv2


def hough_transform(img_bin, theta_res=1, rho_res=1):
    nR, nC = img_bin.shape
    theta = np.linspace(-90.0, 0.0, np.ceil(90.0 / theta_res) + 1.0)
    theta = np.concatenate((theta, -theta[len(theta) - 2::-1]))

    D = np.sqrt((nR - 1) ** 2 + (nC - 1) ** 2)
    q = np.ceil(D / rho_res)
    nrho = 2 * q + 1
    rho = np.linspace(-q * rho_res, q * rho_res, nrho)
    H = np.zeros((len(rho), len(theta)))
    for rowIdx in range(nR):
        for colIdx in range(nC):
            if img_bin[rowIdx, colIdx]:
                for thIdx in range(len(theta)):
                    rhoVal = colIdx * np.cos(theta[thIdx] * np.pi / 180.0) + \
                             rowIdx * np.sin(theta[thIdx] * np.pi / 180)
                    rhoIdx = np.nonzero(np.abs(rho - rhoVal) == np.min(np.abs(rho - rhoVal)))[0]
                    H[rhoIdx[0], thIdx] += 1
    return rho, theta, H


image = cv2.imread("/home/varunbhat/workspace/ml_project/iamDB/data/words/a01/a01-000u/a01-000u-01-01.png", 0)
cv2.threshold(image, 170, 200, cv2.THRESH_BINARY, image)

ro, theta, h = hough_transform(image)

print(max(h.dtype('int').max()))
