import numpy as np
import cv2

from lib.misc import *

def gaussian_filter(img, kernel_size=3, sigma=1.3):
    img = img.astype(np.float32)
    H, W, C = img.shape

    padH = kernel_size // 2
    padW = kernel_size // 2

    img = np.pad(img, [[padH, padH], [padW, padW], [0, 0]], "constant")
   
    K = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    for y in range(-padH, -padH+kernel_size):
        for x in range(-padW, -padW+kernel_size):
            K[y, x] = np.exp(-(x**2 + y**2) / (2*(sigma**2)))
    K /= (2 * np.pi * sigma * sigma)
    K /= K.sum()

    out = np.zeros_like(img)
    newH, newW, newC = out.shape
    for y in range(newH-2*padH):
        for x in range(newW-2*padW):
            for c in range(newC):
                #print(img[y:y+kernel_size, x:x+kernel_size, c].shape, K.shape)
                out[y + padH, x + padW, c] = np.sum(
                    K * img[y:y+kernel_size, x:x+kernel_size, c])

    out = np.clip(out, 0, 255)
    out = out[padH:padH+H, padW:padW+W, :].astype(np.uint8)

    return out

def median_filter(img, kernel_size=3):
    img = img.astype(np.float32)
    H, W, C = img.shape

    padH = kernel_size // 2
    padW = kernel_size // 2

    img = np.pad(img, [[padH, padH], [padW, padW], [0, 0]], "constant")
    
    out = np.zeros_like(img, dtype=np.float32)
    newH, newW, newC = out.shape
    for y in range(newH-2*padH):
        for x in range(newW-2*padW):
            for c in range(newC):
                out[y+padH, x+padW, c] = np.median(img[y:y+kernel_size, x:x+kernel_size, c])

    out = out[padH:padH+H, padW:padW+W, :].astype(np.uint8)
    return out

def smoothing_filter(img, kernel_size=3):
    img = img.astype(np.float32)
    H, W, C = img.shape

    padH = kernel_size // 2
    padW = kernel_size // 2

    img = np.pad(img, [[padH, padH], [padW, padW], [0, 0]], "constant")
    
    out = np.zeros_like(img, dtype=np.float32)
    newH, newW, newC = out.shape
    for y in range(newH-2*padH):
        for x in range(newW-2*padW):
            for c in range(newC):
                out[y+padH, x+padW, c] = np.mean(img[y:y+kernel_size, x:x+kernel_size, c])

    out = out[padH:padH+H, padW:padW+W, :].astype(np.uint8)
    return out

def motion_filter(img, kernel_size=3):
    img = img.astype(np.float32)
    H, W, C = img.shape

    pad = kernel_size // 2

    img = np.pad(img, [[pad, pad], [pad, pad], [0, 0]], "constant")

    K = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    for i in range(kernel_size):
        K[i, i] = 1 / kernel_size

    out = np.zeros_like(img, dtype=np.float32)
    newH, newW, newC = out.shape
    for y in range(newH - 2*pad):
        for x in range(newW - 2*pad):
            for c in range(newC):
                out[y+pad, x+pad, c] = np.sum(K * img[y:y+kernel_size, x:x+kernel_size, c])

    out = out[pad:pad+H, pad:pad+W, :].astype(np.uint8)
    return out

def max_min_filter(img, kernel_size=3):
    img = img.astype(np.float32)
    H, W, C = img.shape

    pad = kernel_size // 2

    img = np.pad(img, [[pad, pad], [pad, pad], [0, 0]], "constant")
    
    out = np.zeros_like(img, dtype=np.float32)
    newH, newW, newC = out.shape
    for y in range(newH - 2*pad):
        for x in range(newW - 2*pad):
            for c in range(newC):
                window = img[y:y+kernel_size, x:x+kernel_size, c]
                out[y+pad, x+pad, c] = np.max(window) - np.min(window)
    
    out = out[pad:pad+H, pad:pad+W, :].astype(np.uint8)
    return out

def differential_filter(img, mode="v"):
    kernel_size = 3
    img = img.astype(np.float32)
    H, W, C = img.shape

    pad = kernel_size // 2

    img = np.pad(img, [[pad, pad], [pad, pad], [0, 0]], "constant")
    out = np.zeros_like(img, dtype=np.float32)

    if mode == "v":
        K = np.asarray([
            [0, -1, 0],
            [0, 1, 0],
            [0, 0, 0]], dtype=np.float32)
    elif mode == "h":
        K = np.asarray([
            [0, 0, 0],
            [-1, 1, 0],
            [0, 0, 0]], dtype=np.float32)
    else:
        raise ValueError(f"Unknown mode '{mode}'. mode is 'v' or 'h'.")

    newH, newW, newC = out.shape
    for y in range(newH-2*pad):
        for x in range(newW-2*pad):
            for c in range(newC):
                out[y+pad, x+pad, c] = abs(np.sum(K * img[y:y+kernel_size, x:x+kernel_size, c]))

    out = out[pad:pad+H, pad:pad+W, :].astype(np.uint8)
    return out

def sobel_filter(img, mode="v"):
    kernel_size = 3

    if img.ndim == 2:
        img = np.expand_dims(img, 2) 
    img = img.astype(np.float32)
    H, W, C = img.shape
    
    pad = kernel_size // 2

    img = np.pad(img, [[pad, pad], [pad, pad], [0, 0 ]], "constant")
    out = np.zeros_like(img, dtype=np.float32)

    if mode == "v":
        K = np.asarray([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]], dtype=np.float32)
    elif mode == "h":
        K = np.asarray([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]], dtype=np.float32)
    else:
        raise ValueError(f"Unknown mode '{mode}'. mode is 'v' or 'h'.")

    newH, newW, newC = out.shape
    for y in range(newH-2*pad):
        for x in range(newW-2*pad):
            for c in range(newC):
                out[y+pad, x+pad, c] = abs(np.sum(K * img[y:y+kernel_size, x:x+kernel_size, c]))

    out = out[pad:pad+H, pad:pad+W, :].astype(np.uint8)
    return out

def prewitt_filter(img):
    kernel_size = 3

    if img.ndim != 2:
        img = bgr2gray(img)        
    img = np.expand_dims(img, 2)
    img = img.astype(np.float32)
    H, W, C = img.shape
    
    pad = kernel_size // 2

    img = np.pad(img, [[pad, pad], [pad, pad], [0, 0]], "constant")
    out = np.zeros_like(img, dtype=np.float32)

    Kv = np.asarray([
        [-1, -1, -1],
        [0, 0, 0],
        [1, 1, 1]], dtype=np.float32)

    Kh = np.asarray([
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]], dtype=np.float32)

    newH, newW, newC = out.shape
    for y in range(newH-2*pad):
        for x in range(newW-2*pad):
            for c in range(newC):
                window = img[y:y+kernel_size, x:x+kernel_size, c]
                dv = np.sum(Kv * window)
                dh = np.sum(Kh * window)
                out[y+pad, x+pad, c] = np.sqrt(dv**2 + dh**2)
    
    out = out[pad:pad+H, pad:pad+W, :].astype(np.uint8)
    return out

def laplacian_filter(img):
    kernel_size = 3

    if img.ndim != 2:
        img = bgr2gray(img)
    img = np.expand_dims(img, 2)
    img = img.astype(np.float32)
    H, W, C = img.shape

    pad = kernel_size // 2
    img = np.pad(img, [[pad, pad], [pad, pad], [0, 0]], "constant")
    out = np.zeros_like(img, dtype=np.float32)

    K = np.asarray([
        [1, 1, 1],
        [1, -8, 1],
        [1, 1, 1]], dtype=np.float32)
    
    newH, newW, newC = out.shape
    for y in range(newH-2*pad):
        for x in range(newW-2*pad):
            for c in range(newC):
                out[y+pad, x+pad, c] = np.sum(
                    K * img[y:y+kernel_size, x:x+kernel_size, c])

    out = np.clip(out, 0, 255)
    out = out[pad:pad+H, pad:pad+W, :].astype(np.uint8)
    return out

def emboss_filter(img):
    kernel_size = 3

    if img.ndim != 2:
        img = bgr2gray(img)
    img = np.expand_dims(img, 2)
    img = img.astype(np.float32)
    H, W, C = img.shape

    pad = kernel_size // 2
    img = np.pad(img, [[pad, pad], [pad, pad], [0, 0]], "constant")
    out = np.zeros_like(img, dtype=np.float32)

    K = np.asarray([
        [-2, -1, 0],
        [-1, 1, 1],
        [0, 1, 2]], dtype=np.float32)
    
    newH, newW, newC = out.shape
    for y in range(newH-2*pad):
        for x in range(newW-2*pad):
            for c in range(newC):
                out[y+pad, x+pad, c] = np.sum(
                    K * img[y:y+kernel_size, x:x+kernel_size, c])

    out = np.clip(out, 0, 255)
    out = out[pad:pad+H, pad:pad+W, :].astype(np.uint8)
    return out

def laplacian_of_gaussian_filter(img, kernel_size=5, sigma=3):
    if img.ndim != 2:
        img = bgr2gray(img)
    img = np.expand_dims(img, 2)
    img = img.astype(np.float32)
    H, W, C = img.shape

    pad = kernel_size // 2
    img = np.pad(img, [[pad, pad], [pad, pad], [0, 0]], "constant")

    out = np.zeros_like(img, dtype=np.float32)

    K = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    for i in range(-pad, -pad+kernel_size):
        for j in range(-pad, -pad+kernel_size):
            K[i, j] = (i**2 + j**2 - sigma**2) * np.exp(-(i**2 + j**2) / (2 * sigma**2))

    K /= 2 * np.pi * (sigma ** 6)
    K /= K.sum()
   
    newH, newW, newC = out.shape
    for i in range(0, newH-2*pad):
        for j in range(0, newW-2*pad):
            for k in range(0, newC):
                out[i+pad, j+pad, k] = np.sum(
                        K * img[i:i+kernel_size, j:j+kernel_size, k])
   
    out = np.clip(out, 0, 255) 
    out = out[pad:H+pad, pad:W+pad, :].astype(np.uint8)
    return out

