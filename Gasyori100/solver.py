import os

import cv2
import numpy as np

from lib.misc import *
from lib.filter import *
from lib.histogram import *

class Solver:
    
    def __init__(self, imori_path, imori_noise_path, imori_dark_path, imori_gamma_path):
        assert os.path.exists(imori_path), f"No such file exists: '{imori_path}'"
        assert os.path.exists(imori_noise_path), f"No such file exists: '{imori_noise_path}'"
        assert os.path.exists(imori_dark_path), f"No such file exists: '{imori_dark_path}'"
        assert os.path.exists(imori_dark_path), f"No such file exists: '{imori_gamma_path}'"
        self.imori_path = imori_path
        self.imori_noise_path = imori_noise_path
        self.imori_dark_path = imori_dark_path
        self.imori_gamma_path = imori_gamma_path

    def solve(self, question):
        if question == 7:
            img = cv2.imread(self.imori_path)

        elif question == 9:
            """ ガウシアンフィルタ """
            img = cv2.imread(self.imori_noise_path)
            out = gaussian_filter(img, kernel_size=3, sigma=1.3)
            out = np.hstack((img, out))
            show_image(out)

        elif question == 10:
            """ メディアンフィルタ """
            img = cv2.imread(self.imori_noise_path)
            out = median_filter(img)
            out = np.hstack((img, out))
            show_image(out)

        elif question == 11:
            """ 平滑化フィルタ """
            img = cv2.imread(self.imori_path)
            out = smoothing_filter(img)
            out = np.hstack((img, out))
            show_image(out)
    
        elif question == 12:
            """ モーションフィルタ """
            img = cv2.imread(self.imori_path)
            out = motion_filter(img)
            out = np.hstack((img, out))
            show_image(out)

        elif question == 13:
            """ MaxMinフィルタ """
            img = cv2.imread(self.imori_path)
            out = max_min_filter(img)
            out = np.hstack((img, out))
            show_image(out)

        elif question == 14:
            """ 微分フィルタ """
            img = cv2.imread(self.imori_path)
            outh = differential_filter(img, mode="h")
            outv = differential_filter(img, mode="v")
            out = np.hstack((img, outh, outv))
            show_image(out, title="original, horizontal, vertical")

        elif question == 15:
            """ Sobelフィルタ """
            img = cv2.imread(self.imori_path)
            gray = bgr2gray(img)
            outv = sobel_filter(img, mode="v")
            outh = sobel_filter(img, mode="h")
            grayv = sobel_filter(gray, mode="v")
            grayh = sobel_filter(gray, mode="h")
            out = np.hstack((img, outv, outh))
            out_gray = np.hstack((grayv, grayh))
            show_image(out, title="Original, SobelFilter(vertical), SobelFilter(horizontal)")
            show_image(out_gray, title="Gray SobelFilter(vertical), SovelFilter(horizontal)")

        elif question == 16:
            """ Prewitt """
            img = cv2.imread(self.imori_path)
            gray = bgr2gray(img)
            out = prewitt_filter(gray)
            show_image(out, title="Prewitt Filter")

        elif question == 17:
            """ Laplacianフィルタ """
            img = cv2.imread(self.imori_path)
            gray = bgr2gray(img)
            out = laplacian_filter(gray)
            show_image(out, title="Laplacian Filter")

        elif question == 18:
            """ Embossフィルタ"""
            img = cv2.imread(self.imori_path)
            gray = bgr2gray(img) 
            out = emboss_filter(gray)
            show_image(out, title="Emboss Filter")

        elif question == 19:
            """ LoGフィルタ """
            img = cv2.imread(self.imori_noise_path)
            gray = bgr2gray(img)
            out = laplacian_of_gaussian_filter(gray)
            show_image(out)

        elif question == 20:
            """ ヒストグラム表示 """
            img = cv2.imread(self.imori_dark_path)
            show_hist(img)

        elif question == 21:
            """ ヒストグラム正規化 """
            img = cv2.imread(self.imori_dark_path)
            out = histogram_normalization(img)
            show_image(out)
            show_hist(out)

        elif question == 22:
            """ ヒストグラム操作 """
            img = cv2.imread(self.imori_dark_path)
            out = change_histogram(img)
            show_image(out)
            show_hist(out)

        elif question == 23:
            """ ヒストグラム平坦化 """
            img = cv2.imread(self.imori_path)
            out = histogram_equalization(img)
            show_image(out)
            show_hist(out)

        elif question == 24:
            """ ガンマ補正 """
            img = cv2.imread(self.imori_gamma_path)
            show_image(img)
            out = gamma_correction(img)
            show_image(out)

        elif question == 25:
            """ 最近傍補間 """
            img = cv2.imread(self.imori_path)
            out = rescale_image_nn()

        else:
            raise ValueError("Unknown question number.")
        

