import cv2
import numpy as np
import matplotlib.pyplot as plot


class HandwritingRecognition:
    def __init__(self, image_path):
        self.debug = False
        self.read_image(image_path)
        temp_image = self.normalize()
        y_segments = self.segment_lines(temp_image)
        x_segments = []

        for s, e in y_segments:
            # self.show_image(cv2.resize(temp_image[s:e, :] * 255, (0, 0), fx=0.5, fy=0.5))
            # print("length:", temp_image.shape)
            x_segments.append(self.segment_words(temp_image[s:e, :]))

        self.segments = []
        for line in range(len(x_segments)):
            self.segments.append([(y_segments[line], (x_s, x_e)) for x_s, x_e in x_segments[line]])

    def normalize(self, image=None):
        t_image = image[:] if image is not None else self.image[:]
        cv2.threshold(t_image, 160, 230, cv2.THRESH_BINARY, t_image)
        cv2.normalize(t_image, t_image, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        return t_image

    def segment_lines(self, image=None, debug=False):
        start_image = image[:] if image is not None else self.image[:]
        t_image = start_image[:]

        # self.show_image(t_image * 255)

        #  Get the histogram of the data
        # print("shape_timage:", t_image.shape)
        data = (np.dot(t_image, np.ones(t_image.shape[1])) / t_image.shape[1]) < 0.98
        data1 = (np.dot(t_image, np.ones(t_image.shape[1])) / t_image.shape[1])

        if debug:
            fig, plt = plot.subplots(1, 1)
            plt.set_ylim(-1, 2)
            plt.set_xlim(0, t_image.shape[1])
            plt.plot(range(0, data1.size), data1)
            plot.show()
            # image = np.array([t_image[:, i].T * data for i in range(t_image.shape[1])]).T
            # self.show_image(cv2.resize(image.T * 255, (0, 0), fx=0.5, fy=0.5))

        start = 0
        segment = []

        for i in range(t_image.shape[1]):
            if data[i - 1] == 0 and data[i] == 1:
                start = i
            elif data[i - 1] == 1 and data[i] == 0:
                segment.append((start, i))
                # self.show_image(t_image[start:i, :].T * 255)

        if debug:
            for s, e in segment:
                # image = cv2.blur(start_image[s - 5:e + 6, :] * 255, (5, 5))
                # self.show_image(image)
                pass

        return segment

    def segment_words(self, image, debug=False):
        # self.show_image(image.T * 255)
        # return self.segment_lines(image[:].T)
        start_image = image[:] if image is not None else self.image[:]
        t_image = start_image[:]

        # self.show_image(t_image * 255)

        #  Get the histogram of the data
        # print("shape_timage:", t_image.shape)
        data = (np.dot(t_image.T, np.ones(t_image.shape[0])) / t_image.shape[0]) < 0.98
        data1 = (np.dot(t_image.T, np.ones(t_image.shape[0])) / t_image.shape[0])

        if debug:
            fig, plt = plot.subplots(1, 1)
            plt.set_ylim(-1, 2)
            plt.set_xlim(0, t_image.shape[1])
            plt.plot(range(0, data1.size), data1)
            plot.show()
            # image = np.array([t_image[:, i].T * data for i in range(t_image.shape[1])]).T
            # self.show_image(cv2.resize(image.T * 255, (0, 0), fx=0.5, fy=0.5))

        start = 0
        segment = []

        for i in range(t_image.shape[1]):
            if data[i - 1] == 0 and data[i] == 1:
                start = i
            elif data[i - 1] == 1 and data[i] == 0:
                segment.append((start, i))
                # self.show_image(t_image[start:i, :].T * 255)

        if debug:
            for s, e in segment:
                # image = cv2.blur(start_image[s - 5:e + 6, :] * 255, (5, 5))
                # self.show_image(image)
                pass
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
                self.show_image(self.image[y_s:y_e, x_s:x_e] * 255)

    def get_segmented_image_array(self):
        arr = []
        for line in self.segments:
            for ((y_s, y_e), (x_s, x_e)) in line:
                arr.append(self.image[y_s:y_e, x_s:x_e])
        return arr


if __name__ == '__main__':

    for
    hreco = HandwritingRecognition('/home/varunbhat/workspace/ml_project/iamDB/data/forms/a01-007u.png')
    count = 1;
    for img in hreco.get_segmented_image_array():
        cv2.imwrite('segmented/%d.png' % count, img * 255)
        count += 1
