import tensorflow as tf
import numpy as np
import cv2

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=False)

class Annotated_MNIST():

    def __init__(self, train=False):
        self.train = train

        if self.train:
            self.labels = mnist.train.labels
            self.images = np.reshape(mnist.train.images, [-1, 28, 28])
        else:
            self.labels = mnist.test.labels
            self.images = np.reshape(mnist.test.images, [-1, 28, 28])


    def get_nums(self, num):
        idx = np.where(self.labels == num)[0]
        nums = self.images[idx]
        return idx, nums


    def thickness_stats(self, num, line, thickness_range):
        idx, nums = self.get_nums(num)

        m = int(np.mean(np.sum(nums[:, line, :] != 0, axis=1)))
        t_lb = int(m - thickness_range / 2)
        t_ub = int(m + thickness_range / 2)

        l_idx = np.where(np.sum(nums[:, line, :] != 0, axis=1) < t_lb)
        n_idx = np.where(np.logical_and(np.sum(nums[:, line, :] != 0, axis=1) >= t_lb, np.sum(nums[:, line, :] != 0, axis=1) <= t_ub))
        h_idx = np.where(np.sum(nums[:, line, :] != 0, axis=1) > t_ub)

        l_idx = idx[l_idx]
        n_idx = idx[n_idx]
        h_idx = idx[h_idx]

        print('Low: {}\nNormal: {}\nHigh: {}\n'.format(len(l_idx), len(n_idx), len(h_idx)))

        return l_idx, n_idx, h_idx


    def rotate_and_scale(self, img, angle, scale):
        M = cv2.getRotationMatrix2D((14, 14), angle, scale)
        return cv2.warpAffine(img, M, (28, 28))


    def skew_stats(self, num, bound, skew_range):

        lb = int(14 - bound / 2)
        ub = int(14 + bound / 2)
        s_lb = -int(skew_range / 2)
        s_ub = int(skew_range / 2)

        idx, nums = self.get_nums(num)

        skews = []
        for num in nums:

            li = []
            max_overlap = 0
            max_angle = 0
            for angle in range(-90, 90, 5):
                temp1 = self.rotate_and_scale(num, angle, 1)
                temp2 = temp1[:, lb:ub]

                if max_overlap <= np.sum(temp2 != 0):
                    max_overlap = np.sum(temp2 != 0)
                    max_angle = angle

            skews.append(max_angle)

        skews = np.array(skews)
        l_idx = np.where(skews < s_lb)
        n_idx = np.where(np.logical_and(skews >= s_lb, skews <= s_ub))
        h_idx = np.where(skews > s_ub)

        l_idx = idx[l_idx]
        n_idx = idx[n_idx]
        h_idx = idx[h_idx]

        print('Low: {}\nNormal: {}\nHigh: {}\n'.format(len(l_idx), len(n_idx), len(h_idx)))

        return l_idx, n_idx, h_idx


