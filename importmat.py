import os

import numpy as np
import cv2
import scipy.io as sio

# get image using CV
# img = cv2.imread("/home/dumma/Desktop/Machine_Learning/Project/English/Hnd/Img/Sample001/img001-001.png", 0)
# cv2.imshow('cv2.WINDOW_NORMAL',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# print(img)


# get image from matlab
mat = sio.loadmat('lists.mat', mat_dtype=True, squeeze_me=False, chars_as_strings=True)

if not os.path.exists('mat_decode'):
    os.mkdir('mat_decode')

for i in range(10):
    if mat['list'][0, 0][i].dtype == '<U34' or mat['list'][0, 0][i].dtype == '<U21':
        np.savetxt('mat_decode/%d.txt' % i, mat['list'][0, 0][i], fmt='%s')
    else:
        np.savetxt('mat_decode/%d.txt' % i, mat['list'][0, 0][i])

print(((mat['list'])[0,0])[1])

# open('mat_decode/%d' % i, 'w').write(mat['list'][0, 0][i])
# struc = mat['list']
# print(struc(0))
