import cv2

def rgb2bgr(img):
    return img[..., ::-1]

def show_image(img, title="Image", rgb=False):
    if rgb:
        img = rgb2bgr(img)
    cv2.imshow(title, img)
    cv2.waitKey(0)

