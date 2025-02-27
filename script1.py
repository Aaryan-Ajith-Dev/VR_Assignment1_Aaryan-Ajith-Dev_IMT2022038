import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
img = cv2.imread('coins.jpeg')

# perform normalisation
img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

# convert to gray scale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# # remove noise using Gaussian filter
gray = cv2.GaussianBlur(gray, (5, 5), 0)


def part1(img, gray):
    # Apply Gaussian Blur to reduce noise

    edges = cv2.Canny(gray, 50, 100)

    # Display the result
    plt.figure(figsize=(8, 6))
    plt.imshow(edges, cmap='gray')
    plt.title("Canny Edge Detection")
    plt.axis("off")
    plt.show()

    return edges  # Return edges for later processing


def part2_3(img, gray):
    # apply thresholding
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.bitwise_not(thresh, thresh)

    # apply morphological operations
    kernel = np.ones((9, 9), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # display the thresholded image
    plt.figure(figsize=(10, 5))
    plt.imshow(thresh, cmap='gray')
    plt.title('Thresholded Image')
    plt.show()

    # find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # filter contours based on area
    contours = [c for c in contours if cv2.contourArea(c) > 10]

    # create a mask for each detected coin
    masks = []
    for c in contours:
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [c], -1, 255, thickness=cv2.FILLED)
        masks.append(mask)
    
    # display the masks for all coins together
    plt.figure(figsize=(10, 5))

    for i, mask in enumerate(masks):
        plt.subplot(1, len(masks), i + 1)
        plt.imshow(mask, cmap='gray')
        plt.title(f'Coin {i + 1}')
        plt.axis('off')

    plt.show()

    print("Number of coins detected:",len(contours))

# def hough_circles(img, gray):
#     # Apply Hough Transform to detect circles
#     gray1 = cv2.GaussianBlur(gray, (15, 15), 0)
#     circles = cv2.HoughCircles(gray1, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)
#     circles = np.uint16(np.around(circles))

#     # draw the detected circles
#     for i in circles[0, :]:
#         # draw the outer circle
#         cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
#         # draw the center of the circle
#         cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)

#     # display the image with detected circles
#     plt.figure(figsize=(10, 5))
#     plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#     plt.title('Detected Circles via Hough Transform')
#     plt.show()

#     print("Number of coins (using Hough Circles)",len(circles[0]))

part1(img.copy(), gray.copy())
part2_3(img.copy(), gray.copy())



