import cv2
import numpy as np
import matplotlib.pyplot as plt

def distance_transform(img):
    # Compute the normalized distance from the boundary of image 'img'
    dist = np.zeros_like(img, dtype=np.float32)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            dist[i, j] = min(i, j, img.shape[0] - i, img.shape[1] - j)
    
    normalized_dist = dist / np.max(dist)
    return normalized_dist

def blend(img1, img2, alpha1, alpha2):
    # Blend the images img1 and img2 using the alpha masks alpha1 and alpha2 as weights
    blended = img1.copy()
    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            blended[i, j] = (alpha1[i, j] * img1[i, j] + alpha2[i, j] * img2[i, j]) / (alpha1[i, j] + alpha2[i, j])
    return blended

def create_panorama(image1_path, image2_path, output_path):
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)

    # Convert to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # SIFT feature detector
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    # BFMatcher with ratio test
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # Extract matched keypoints
    points1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    points2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

    # here m.queryIdx gives the index of the descriptor in the first set of descriptors (des1) for the current match
    # and m.trainIdx gives the index of the descriptor in the second set of descriptors (des2) for the current match
    


    # Compute homography
    h, mask = cv2.findHomography(points2, points1, cv2.RANSAC)


    # find distance transform for both images
    alpha1 = distance_transform(gray1)
    alpha2 = distance_transform(gray2)

    # Warp the second image
    height, width, _ = img1.shape
    panorama = cv2.warpPerspective(img2, h, (width * 3, height))

    # find the warped distance transform
    warped_alpha2 = cv2.warpPerspective(alpha2, h, (width * 3, height))

    # find the warped distance transform for the first image
    warped_alpha1 = np.zeros_like(warped_alpha2)
    warped_alpha1[:height, :width] = alpha1

    print(panorama.shape, width, img2.shape)

    # Create masks for the images
    img1_mask = np.zeros_like(panorama)
    img1_mask[:height, :width] = img1
    img2_mask = panorama.copy()

    # Blend the images
    print(img1_mask.shape, img2_mask.shape, warped_alpha1.shape, warped_alpha2.shape)
    blended = blend(img1_mask, img2_mask, warped_alpha1, warped_alpha2)
    
    # panorama[0:height, 0:width] = img1

    plt.imshow(alpha1), plt.show()
    plt.imshow(warped_alpha2), plt.show()
    
    # Save the blended panorama
    cv2.imwrite(output_path, blended)
    plt.imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()




create_panorama('middle.jpeg', 'right.jpeg', 'output.jpeg')
create_panorama('left.jpeg', 'output.jpeg', 'final.jpeg')
