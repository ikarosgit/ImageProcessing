import os

import cv2

from lib.misc import show_image

class Solver:
    
    def __init__(self, imori_path, imori_noise_path):
        assert os.path.exists(imori_path), f"No such file Exists: '{imori_path}'"
        assert os.path.exists(imori_noise_path), f"No such file Exists: '{imori_noise_path}'"
        self.imori_path = imori_path
        self.imori_noise_path = imori_noise_path

    def solve(self, question):
        if question == 7:
            img = cv2.imread(self.imori_path)
        elif question == 11:
            img = cv2.imread(self.imori_path)

            show_image(img)
        else:
            raise ValueError("Unknown question number.")
        

