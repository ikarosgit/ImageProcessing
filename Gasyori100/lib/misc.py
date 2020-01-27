import numpy as np
import cv2
import matplotlib.pyplot as plt

def rgb2bgr(img):
    return img[..., ::-1]

def bgr2gray(img):
    """
    Y = 0.2126 R + 0.7152 G + 0.0722 B
    """
    img = img.astype(np.float32)
    out = 0.2126*img[:, :, 2] + 0.7152*img[:, :, 1] + 0.0722*img[:, :, 0]
    return out.astype(np.uint8)

def show_image(img, title="Image", rgb=False):
    if rgb:
        img = rgb2bgr(img)
    cv2.imshow(title, img)
    cv2.waitKey(0)

def show_hist(img, bins=255, title="Histgram"):
    flatten = img.ravel().astype(np.float32)
    plt.hist(flatten, bins=bins, rwidth=0.8, range=(0, 255))
    plt.xlabel("Value")
    plt.ylabel("Count")
    plt.title(title)
    plt.show()

def gamma_correction(img, c=1, g=2.2):
    img_shape = img.shape
    out = img.copy().astype(np.float32)
    out /= 255.
    out = (out / c) ** (1 / g)
    out *= 255.
    out = out.astype(np.uint8)
    return out.reshape(*img_shape)
