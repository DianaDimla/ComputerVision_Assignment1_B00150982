import cv2 as cv
import numpy as np
import time

from collections import deque # For BFS in connected components

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

def closing(img):
    dilated = dilation(img)
    closed = erosion(dilated)
    return closed

# Connected Components using BFS
def connected_components(binary):

    labels = np.zeros(binary.shape, dtype=int)
    current_label = 1

    rows, cols = binary.shape

    for x in range(rows):
        for y in range(cols):
            
            # If pixel is foreground and not labbelled
            if binary[x, y] == 255 and labels[x, y] == 0:

                # Start BFS
                queue = deque()
                queue.append((x, y))
                labels[x, y] = current_label

                while queue:
                    cx, cy = queue.popleft()

                    # 4-connected neighbors
                    neighbors = [(cx-1, cy), (cx+1, cy), (cx, cy-1), (cx, cy+1)]

                    for nx, ny in neighbors:
                        if 0 <= nx < rows and 0 <= ny < cols:
                            if binary[nx, ny] == 255 and labels[nx, ny] == 0:
                                labels[nx, ny] = current_label
                                queue.append((nx, ny))

                current_label += 1
    return labels

# Extract largest component
def extract_largest_component(labels):

    unique, counts = np.unique(labels, return_counts=True)

    # Remove background label (0)    
    counts = counts[unique != 0]
    unique = unique[unique != 0]

    if len(counts) == 0:
        return None
    
    largest_label = unique[np.argmax(counts)]

    mask = np.zeros(labels.shape, dtype=np.uint8)
    mask[labels == largest_label] = 255

    return mask


# Classify O-ring based on extracted region area
def classify_ring(ring):

    if ring is None:
        return "FAIL", 0

    area = np.sum(ring == 255)

    if 22000 <= area <= 34000:
        result = "PASS"
    else:
        result = "FAIL"

    return result, area


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

    # Connected Components and extract largest
    labels = connected_components(bw)
    ring = extract_largest_component(labels)
    result, area = classify_ring(ring)

    # Display results
    if ring is not None:
        cv.imshow('Labels', ring)

    cv.putText(rgb, "Image: " + str(i), (20, 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv.putText(rgb, "Time: " + str(round(end - start, 2)) + "s", (20, 55), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    #Green if PASS, Red if FAIL
    color = (0, 255, 0) if result == "PASS" else (0, 0, 255)
    cv.putText(rgb, "Result: " + result, (20, 80), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv.imshow('Result', rgb)
    cv.waitKey(0)

cv.destroyAllWindows()