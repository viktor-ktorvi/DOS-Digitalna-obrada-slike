import numpy as np


class Sample:
    def __init__(self, img, height, width, threshold):
        self.threshold = threshold
        self.height = height
        self.width = width
        self.img = img

        self.sample = self.img[self.height[0]: self.height[1], self.width[0]: self.width[1], :]
        # self.sample = self.sample / np.amax(self.sample)
        self.X = self.sample.reshape((self.sample.shape[0] * self.sample.shape[1], 3)).T

        sigma = np.cov(self.X)
        self.M = np.mean(self.X, 1)
        self.sigma_inv = np.linalg.inv(sigma)
