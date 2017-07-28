import tensorflow as tf
import numpy as np

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


    def thickness_stats(self, num, line):
        idx, nums = self.get_nums(num)

        m = int(np.mean(np.sum(nums[:, line, :] != 0, axis=1)))
        l_idx = np.where(np.sum(nums[:, line, :] != 0, axis=1) < m)[0]
        n_idx = np.where(np.sum(nums[:, line, :] != 0, axis=1) == m)[0]
        h_idx = np.where(np.sum(nums[:, line, :] != 0, axis=1) > m)[0]

        l = nums[l_idx]
        n = nums[n_idx]
        h = nums[h_idx]

        l_idx = idx[l_idx]
        n_idx = idx[n_idx]
        h_idx = idx[h_idx]

        print('\nLow: {}\nNormal: {}\nHigh: {}'.format(len(l), len(n), len(h)))

        return l, l_idx, n, n_idx, h, h_idx

