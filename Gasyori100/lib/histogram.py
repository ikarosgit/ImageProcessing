import numpy as np

def histogram_normalization(img, min_value=0, max_value=255):
    H, W, C = img.shape
    img = img.astype(np.float32)
    c = np.min(img)
    d = np.max(img)
    out = np.zeros_like(img, dtype=np.uint8)
    out[img < min_value] = min_value
    out[img > max_value] = max_value
    _range = np.logical_and(min_value <= img, img < max_value)
    out[_range] = (img[_range] - c) * (max_value - min_value) / (d - c) + min_value
    out = np.clip(out, 0, 255).astype(np.uint8)
    out = out.reshape(H, W, C)
    return out

def change_histogram(img, mean=128., std=52.):
    img_shape = img.shape
    img = img.ravel().astype(np.float32)
    img_mean = np.mean(img)
    img_std = np.std(img)

    out = np.zeros_like(img, dtype=np.float32)
    out = std * (img - img_mean) / img_std + mean
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out.reshape(*img_shape)

def histogram_equalization(img, z_max=255):
    img_shape = img.shape
    img = img.ravel().astype(np.float32)
    S = len(img)
    out = img.copy()

    sum_h = 0

    for i in range(1, 255):
        index = np.where(img == i)
        sum_h += len(img[index])
        z_prime = z_max / S * sum_h
        out[index] = z_prime

    out = np.clip(out, 0, 255).astype(np.uint8)
    return out.reshape(*img_shape)
