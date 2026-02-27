import cv2 as cv
import numpy as np
import time

# Histogram Manual
def histogram(img):
    hist = np.zeros(256, dtype=int)
    for x in range(0, img.shape[0]):
        for y in range(0, img.shape[1]):
            hist[img[x,y]] += 1
    return hist

# Otsu Threshold Method
def otsu_threshold(hist, total_pixels):

    sum_total = 0
    for t in range(256):
        sum_total += t * hist[t]

    sum_background = 0
    weight_background = 0
    max_variance = 0
    best_threshold = 0

    for t in range(256):

        weight_background += hist[t]
        if weight_background == 0:
            continue

        weight_foreground = total_pixels - weight_background
        if weight_foreground == 0:
            break

        sum_background += t * hist[t]

        mean_background = sum_background / weight_background
        mean_foreground = (sum_total - sum_background) / weight_foreground

        between_variance = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2

        if between_variance > max_variance:
            max_variance = between_variance
            best_threshold = t
        
        return best_threshold


# Thresholding
def threshold(img, thresh):

    binary = np.zeros_like(img)

    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            if img[x,y] > thresh:
                binary[x,y] = 255
            else:
                binary[x,y] = 0
    return binary

# Binary Morphology
def dilation(img):

    output = np.zeros_like(img)

    for x in range(1, img.shape[0] - 1):
        for y in range(1, img.shape[1] - 1):
            if np.max(img[x-1:x+2, y-1:y+2]) == 255:
                output[x,y] = 255
    return output

def erosion(img):

    output = np.zeros_like(img)

    for x in range(1, img.shape[0] - 1):
        for y in range(1, img.shape[1] - 1):
            if np.min(img[x-1:x+2, y-1:y+2]) == 255:
                output[x,y] = 255
    return output

def closing (img):
    dilated = dilation(img)
    closed = erosion(dilated)
    return closed

# Main
for i in range(1,16):
    #read in an image into memory
    img = cv.imread('Orings/Oring' + str(i) + '.jpg', 0)
    start = time.time()

    # Otsu Thresholding
    hist = histogram(img)
    thresh = otsu_threshold(hist, img.size)

    bw = threshold(img, thresh)

    # Closing
    bw = closing(bw)
    end = time.time()
    rgb = cv.cvtColor(bw, cv.COLOR_GRAY2BGR)

    cv.putText(rgb, "Image: " + str(i), (20, 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv.putText(rgb, "Time: " + str(round(end - start, 2)) + "s", (20, 55), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    cv.imshow('Binary Result',rgb)
    cv.waitKey(0)

cv.destroyAllWindows()
