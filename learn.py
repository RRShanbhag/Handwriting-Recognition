import os
import re
from queue import Queue
from threading import Thread

import cv2
import math
import numpy as np
import matplotlib.pyplot as plot
import sys

image_path = 'iamDB/data/forms'

thread_queue = Queue(10)


class HandwritingRecognition:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image_id = image_path.split('/')[-1].split('.')[-2]
        self.image = None
        self.debug = False
        self.read_image(image_path)
        self.y_offset = (620, 2800)
        self.x_offset = (0, self.image.shape[1])
        self.segments = []
        ascii_fp = open('/home/varunbhat/workspace/ml_project/iamDB/data/ascii/words.txt')
        self.word_data = ascii_fp.read()
        ascii_fp.close()

        self.dataset_segments = []

    def segment(self):
        temp_image = np.copy(self.image)
        temp_image = temp_image[self.y_offset[0]:self.y_offset[1], self.x_offset[0]:self.x_offset[1]]
        temp_image = cv2.blur(temp_image, (30, 30))
        temp_image = self.normalize(temp_image)
        # self.show_image(cv2.resize(temp_image, (0, 0), fx=0.3, fy=0.3))
        y_segments = self.segment_lines(temp_image)
        x_segments = []
        for s, e in y_segments:
            temp_image_cropped = np.copy(self.image[s:e, :])
            temp_image_cropped = cv2.blur(temp_image_cropped, (50, 200))
            temp_image_cropped = self.normalize(temp_image_cropped)
            x_segments.append(self.segment_words(temp_image_cropped))

        self.segments = []
        for line in range(len(x_segments)):
            self.segments.append([(y_segments[line], (x_s, x_e)) for x_s, x_e in x_segments[line]])

    def normalize(self, image=None):
        t_image = image[:] if image is not None else self.image[:]
        hist = cv2.calcHist([t_image], [0], None, [256], [0, 256]).T[0]
        threshold_val = 0

        # fig, plt = plot.subplots(1, 1)
        # # plt.set_ylim(, 2)
        # plt.set_xlim(150, 255)
        # plt.plot(range(len(hist)), hist)
        # plot.show()

        for i in range(len(hist)):
            if hist[i] > max(hist) * 0.02234:
                threshold_val = i
                break

        # threshold_val = 200
        # print(threshold_val, threshold_val + ((255 - threshold_val) * .75))
        cv2.threshold(t_image, threshold_val, threshold_val + ((255 - threshold_val) * .40), cv2.THRESH_OTSU, t_image)
        cv2.normalize(t_image, t_image, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        # self.show_image(cv2.resize(t_image, (0, 0), fx=0.3, fy=0.3) * 255)
        return t_image

    def segment_lines(self, image=None):
        start_image = image if image is not None else self.image
        t_image = start_image

        #  Get the histogram of the data
        data = (np.dot(t_image, np.ones(t_image.shape[1])) / t_image.shape[1]) < 0.95
        # data1 = (np.dot(t_image, np.ones(t_image.shape[1])) / t_image.shape[1])

        # fig, plt = plot.subplots(1, 1)
        # plt.set_ylim(-1, 2)
        # plt.set_xlim(0, t_image.shape[1])
        # plt.plot(range(len(data)), data)
        # plot.show()

        start = 0
        segment = []

        for i in range(1, t_image.shape[0]):
            if data[i - 1] == 0 and data[i] == 1:
                start = i
            elif data[i - 1] == 1 and data[i] == 0:
                _s, _e = (start + self.y_offset[0], i + self.y_offset[0])
                if _e - _s < 5:
                    continue
                segment.append((start + self.y_offset[0], i + self.y_offset[0]))
                # self.show_image(cv2.resize(self.image[_s:_e, :], (0, 0), fx=0.3, fy=0.3))
        return segment

    def segment_words(self, image):
        start_image = image if image is not None else self.image
        t_image = start_image

        # self.show_image(t_image * 255)
        # print(t_image.shape[0])
        #  Get the histogram of the data
        # if t_image.shape[0] < 10 or t_image.shape[1] < 10:
        #     return []
        data = (np.dot(t_image.T, np.ones(t_image.shape[0])) / t_image.shape[0]) < 0.95
        start = 0
        segment = []

        for i in range(1, t_image.shape[1]):
            if data[i - 1] == 0 and data[i] == 1:
                start = i
            elif data[i - 1] == 1 and data[i] == 0:
                _s, _e = (start + self.x_offset[0], i + self.x_offset[0])
                if _e - _s < 5:
                    continue
                segment.append((start + self.x_offset[0], i + self.x_offset[0]))
        return segment

    def read_image(self, img_path, bw=True):
        if bw:
            self.image = cv2.imread(img_path, 0)
        else:
            self.image = cv2.imread(img_path)

    def show_image(self, image=None, window='image'):
        # show the image specified in the window
        cv2.imshow(window, image if image is not None else self.image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def show_segmented_words(self):
        for line in self.segments:
            for ((y_s, y_e), (x_s, x_e)) in line:
                self.show_image(self.image[y_s:y_e, x_s:x_e])

    def get_segmented_image_array(self):
        arr = []
        for line in self.segments:
            for ((y_s, y_e), (x_s, x_e)) in line:
                arr.append(self.image[y_s - 6:y_e + 6, x_s + 3:x_e + 3])
        return arr

    def read_dataset_segmentation(self):
        rows = re.findall('(%s.*)' % self.image_id, self.word_data)
        rexp = re.compile(
            '(?P<id>[a-z0-9\-]+) (?P<status>err|ok) (?P<threshold>\d+) (?P<coordinates>([\d\-]+ ){4})'
            '(?P<typeset>.*?) (?P<word>.*)')
        for data in rows:
            res = rexp.match(data)
            if res is None:
                continue
            x, y, w, h = (int(_i) for _i in res.group('coordinates').split(' ')[:4])
            self.show_image(self.image[y:y + h, x:x + w])


def segment(pth):
    hreco = HandwritingRecognition(os.path.join(image_path, pth))
    count = 1
    for img in hreco.get_segmented_image_array():
        if not os.path.exists(os.path.join('segmented', pth)):
            os.mkdir(os.path.join('segmented', pth))

        cv2.imwrite(os.path.join('segmented', pth, '%d.png' % count), img)
        count += 1


def read_dataset(path):
    hreco = HandwritingRecognition(os.path.join(image_path, pth))
    hreco.read_dataset_segmentation()


def dispacher():
    while True:
        t = thread_queue.get()
        t.start()


if __name__ == '__main__':
    # Thread(target=dispacher).start()
    for pth in os.listdir(image_path):
        if 'png' not in pth:
            continue
        print(pth)
        # thread_queue.put(Thread(target=segment, args=(pth,)))
        # segment(pth)
        read_dataset(pth)
