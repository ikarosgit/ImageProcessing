import numpy as np
import cv2

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

