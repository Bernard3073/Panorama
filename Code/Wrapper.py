#!/usr/bin/evn python

"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project1: MyAutoPano: Phase 1 Starter Code

Author(s): 
Chahat Deep Singh (chahat@terpmail.umd.edu) 
PhD Student in Computer Science,
University of Maryland, College Park

Nitin J. Sanket (nitinsan@terpmail.umd.edu)
PhD Candidate in Computer Science,
University of Maryland, College Park
"""

# Code starts here:

import numpy as np
import cv2
# Add any python libraries here
import os
import matplotlib.pyplot as plt
import matplotlib.image as img
import random


def imregionalmax(image, kernel=None):
    """Find the regional max of the image. An approximation of MATLAB's
    imregionalmax function. Result only differs when surrounding pixels
    have the same value as the center.

    Parameters:
    - image: the input image
    - kernel: the size of the neiborhood region, default is 3x3, i.e.
              neighboring 8 pixels.
    Returns:
    - a bitmask image, where '1' indicates local maxima.
    Author:
    - Yu Fang
    References:
    - https://github.com/bhardwajvijay/Utils/blob/master/utils.cpp
    - https://stackoverflow.com/questions/5550290/find-local-maxima-in-grayscale-image-using-opencv
    """
    # dialate the image so that small values are replaced by local max
    local_max = cv2.dilate(image, kernel)
    # non-local max pixels (excluding pixel w/ constant 3x3 neighborhood)
    # will be replaced by local max, so the values will increase. remove them.
    # so the result is either local max or constant neighborhood.
    max_mask = image >= local_max
    # erode the image so that high values are replaced by local min
    local_min = cv2.erode(image, kernel)
    # only local min pixels and pixels w/ constant 3x3 neighborhood
    # will stay the same, otherwise pixels will be replaced by the local
    # min and become smaller. We only take non-local min, non-constant values.
    min_mask = image > local_min
    # boolean logic hack
    #   (local max || constant) && (!local min && !constant)
    # = local max && !local min && !constant
    # = local max && !constant
    return (max_mask & min_mask).astype(np.uint8)


def est_homography(src, dest):
    """ Compute the homography matrix from (x_src, y_src) to (x_dest, y_dest).
    Parameters:
    - src: (x,y) coordinates of N source pixels, where coordinates are row vectors,
           so the matrix has dimension Ndest[:, 0] (N>=4).
    - dest: (x,y) coordinates of N destination pixels, where coordinates are row vectors,
            so the matrix has dimension Ndest[:, 0] (N>=4).
    Returns:
    - the homography matrix such that H @ [x_src, y_src, 1].T = [x_dest, y_dest, 1].T
    Author:
    - Yu Fang
    """
    # N = src.shape[0]
    # if N != dest.shape[0]:
    #     raise ValueError("src and diff should have the same dimension")
    # src_h = np.hstack((src, np.ones((N, 1))))
    # A = np.array([np.block([[src_h[n], np.zeros(3), -dest[n, 0] * src_h[n]],
    #                         [np.zeros(3), src_h[n], -dest[n, 1] * src_h[n]]])
    #               for n in range(N)]).reshape(2 * N, 9)
    # [_, _, V] = np.linalg.svd(A)
    # # take the right singular vector x corresponding to the least singular value
    # # s.t. ||Ax - 0||^2 is minimized
    # return V.T[:, 8].reshape(3, 3)
    
    A = []
    for i in range(len(src)):
        src_x, src_y = src[i][0], src[i][1]
        dest_x, dest_y = dest[i][0], dest[i][1]
        A.append([src_x, src_y, 1, 0, 0, 0, -dest_x * src_x, -dest_x * src_y, -dest_x])
        A.append([0, 0, 0, src_x, src_y, 1, -dest_y * src_x, -dest_y * src_y, -dest_y])
    
    A = np.array(A)
    _, _, V_t = np.linalg.svd(A)
    return V_t[-1, :].reshape(3, 3)


def apply_homography(H, src):
    """ Apply the homography H to src
    Parameters:
    - H: the 3x3 homography matrix
    - src: (x,y) coordinates of N source pixels, where coordinates are row vectors,
           so the matrix has dimension Ndest[:, 0] (N>=4).
    Returns:
    - src: (x,y) coordinates of N destination pixels, where coordinates are row vectors,
           so the matrix has dimension Ndest[:, 0] (N>=4).
    Author:
    - Yu Fang
    """
    src_h = np.hstack((src, np.ones((src.shape[0], 1))))
    dest = src_h @ H.T
    return (dest / dest[:, [2]])[:, 0:2]


def drawMatches(image1, kp1, image2, kp2, idx_pairs):
    """A wrapper around OpenCV's drawMatches.

    Parameters:
    - image1: the first image
    - kp1: *matrix indices* of the keypoints from image 1
           (Ndest[:, 0] numpy array, where N is the number of keypoints)
    - image2: the second image
    - kp2: *matrix indices* of the keypoints from image 2 
           (Ndest[:, 0] numpy array, where N is the number of keypoints)
    - idx_pairs: pairs of matching indices, e.g. if kp1[3] 
                 matches kp2[5], then idx_pairs=[[3,5],...]
                 (Kdest[:, 0] numpy array, where K is the number of matches)
    Returns:
    - an image showing matching points
    Author:
    - Yu Fang
    """
    # note that the coordinates are reversed because the difference
    # between matrix indexing & coordinates.
    keypt1 = [cv2.KeyPoint(coord[1], coord[0], 40) for coord in kp1.tolist()]
    keypt2 = [cv2.KeyPoint(coord[1], coord[0], 40) for coord in kp2.tolist()]
    matches = [cv2.DMatch(pair[0], pair[1], 0, _distance=0)
               for pair in idx_pairs.tolist()]
    return cv2.drawMatches(image1, keypt1, image2, keypt2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Displays image
def visualize_image(image):
    plt.figure(figsize=(10, 10))
    plt.imshow(image, interpolation='nearest')
    plt.axis('off')
    plt.show()

# Overlays points on image
def visualize_points(image, points):
    visualize_image(image)
    # A little messy manipulation
    points = points.T
    plt.plot(points[1], points[0], 'r.', markersize=4)
    plt.show()

# # Finding corners using corner harris
def detect_corners(image):
    # return cv2.cornerHarris(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 5, 5, 0.04)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 5, 5, 0.04)
    dst = cv2.dilate(dst, None)
    image[dst > 0.01*dst.max()] = [255, 0, 0]
    # cv2.imshow('dst', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return image

# Efficient ANMS implementation
def anms(c_img):
    # Local maxima
    maxima = np.array(np.where(imregionalmax(c_img) == 1)).T
    x, y = maxima[:, 0], maxima[:, 1]
    # Get corner score of local maxima
    maxima_c_img = c_img[x, y]
    # Number of local maxima
    n_strong = len(x)
    # Keep best 2% of corners
    n_best = int(0.02 * n_strong)
    r = np.zeros(n_strong)
    for i in np.arange(n_strong):
        # Prune edges, only look at possible corners
        if maxima_c_img[i] >= 0.001 * np.max(maxima_c_img):
            # Check whether point is robust
            threshold = maxima[maxima_c_img[i] < 0.9*maxima_c_img]
            # Calculate euclidean distance
            dist = np.linalg.norm(threshold - maxima[i], axis=1)
            if len(dist) != 0:
                # Keep minimum distance only
                r[i] = np.min(dist)
    # Return indices of best corners
    return maxima[np.argsort(-r)][:(n_best-1)]

# Use cv2 SIFT to create 128src[:,0] feature descriptors
def feature_descriptors(image, corners):
    sift = cv2.SIFT_create()
    keypoints = np.array([cv2.KeyPoint(c[1], c[0], 40) for c in corners])
    _, desc = sift.compute(image, keypoints)
    # Normalize feature descriptors
    return (desc - np.mean(desc, axis=0))/np.std(desc, axis=0)

# Match features between two images
def feature_matching(features):
    matches = np.array([np.zeros((2,))])
    # Look at every keypoint in first image
    for d in np.arange(features[0, 1].shape[0]):
        # Calculate similarity of feature descriptors
        dist = np.linalg.norm(features[1, 1] - features[0, 1][d], axis=1)
        best, second_best = tuple(np.argsort(dist)[:2])
        # Ratio of lowest and second lowest distance
        if dist[best]/dist[second_best] < 0.998:
            # Only keep matches under threshold
            matches = np.append(matches, [[d, best]], axis=0).astype(int)
    return matches[1:]

# Refine matches and estimate robust homography with RANSAC
# def ransac(matches, features):
#     n_max = 10000
#     H_best, inliers_best, n_best = None, None, -np.inf
#     for n in np.arange(n_max):
#         # Select 4 random pairs of matches
#         rand = matches[random.sample(range(0, matches.shape[0]), 4)]
#         p1 = features[0, 0][rand[:, 0]]
#         p2 = features[1, 0][rand[:, 1]]
#         # Estimate homography using 4 random pairs of matches
#         H = est_homography(p1, p2)
#         # Estimate points with homography
#         H_p1 = apply_homography(H, features[0, 0][matches[:, 0]])
#         actual = features[1, 0][matches[:, 1]]
#         # Compare estimated points with actual points
#         dist = np.linalg.norm(H_p1 - actual, axis=1)
#         # Ensure difference is around 3 or 4 pixels
#         inliers = matches[dist < 4]
#         # Keep the largest set of inliers and estimated homography
#         if inliers.shape[0] > n_best:
#             n_best = inliers.shape[0]
#             inliers_best = inliers
#             H_best = H
#         # Terminate is 90% of keypoints have been reached
#         if n_best >= 0.9*H_p1.shape[0]:
#             break
#     return H_best, inliers_best

def computeH_norm(x1, x2):
	#Compute the centroid of the points
	n = len(x1)
	x1_cent_x, x1_cent_y = np.sum(x1[:, 0])/n, np.sum(x1[:, 1])/n
	x2_cent_x, x2_cent_y = np.sum(x2[:, 0])/n, np.sum(x2[:, 1])/n

	#Shift the origin of the points to the centroid
	x1_norm = x1 - [x1_cent_x, x1_cent_y]
	x2_norm = x2 - [x2_cent_x, x2_cent_y]

	#Normalize the points so that the largest distance from the origin is equal to sqrt(2)
	scaling_factor_1 = np.sqrt(2) / np.max([i[0]**2 + i[1]**2 for i in x1_norm])
	scaling_factor_2 = np.sqrt(2) / np.max([i[0]**2 + i[1]**2 for i in x2_norm])
	x1_norm = x1_norm * scaling_factor_1
	x2_norm = x2_norm * scaling_factor_2

	#Similarity transform 1
	# normalization in matrix form
	T1 = np.array([[scaling_factor_1, 0, -scaling_factor_1 * x1_cent_x],
		[0, scaling_factor_1, -scaling_factor_1 * x1_cent_y],
		[0, 0, 1]])

	#Similarity transform 2
	T2 = np.array([[scaling_factor_2, 0, -scaling_factor_2 * x2_cent_x],
		[0, scaling_factor_2, -scaling_factor_2 * x2_cent_y],
		[0, 0, 1]])

	#Compute homography
	H = est_homography(x1_norm, x2_norm)

	#Denormalization
	# H = inv(T1) * H_til * T2
	H = np.linalg.inv(T1) @ H @ T2

	return H


def ransac(matches, features):
    N = np.inf
    sample_count = 0
    p = 0.99
    threshold = 60
    max_num_inlier = 0
    f1 = features[0, 0]
    f2 = features[1, 0]
    while N > sample_count:
        # Select 4 random pairs of matches
        rand = matches[random.sample(range(0, matches.shape[0]), 4)]
        rand_p1 = f1[rand[:, 0]]
        rand_p2 = f2[rand[:, 1]]
        # Estimate homography using 4 random pairs of matches
        H = computeH_norm(rand_p1, rand_p2)
        inlier_count = 0
        inliers = []
        for p1, p2 in zip(f1, f2):
            p1_homo = np.append(p1, 1).reshape(3, 1)
            # Estimate points with homography
            p2_est = H @ p1_homo
            p2_est = (p2_est/p2_est[2])[:2].reshape(1, 2)
            if cv2.norm(p2 - p2_est) <= threshold:
                inlier_count += 1
                inliers.append(1)
            else:
                inliers.append(0)

        if inlier_count > max_num_inlier:
            max_num_inlier = inlier_count
            H_best = H
            inliers.append(p1)
        inlier_ratio = inlier_count / len(f1)
        if np.log(1 - (inlier_ratio**8)) == 0:
            continue
        N = np.log(1-p) / np.log(1 - (inlier_ratio**8))
        sample_count += 1

    return H_best, inliers

# Poorly stitch images together
def stitch(image1, image2, H):
    # Find height and width of both images
    x1, y1 = tuple(image1.shape[:2])
    x2, y2 = tuple(image2.shape[:2])
    # Find estimated location of bounding box
    top_left = apply_homography(H, np.array([[0, 0]]))[0]
    top_right = apply_homography(H, np.array([[0, y1]]))[0]
    bottom_left = apply_homography(H, np.array([[x1, 0]]))[0]
    bottom_right = apply_homography(H, np.array([[x1, y1]]))[0]
    # Determine left and right bounds to calculate width of stitched image
    left_bound = np.amin([top_left[0], bottom_left[0], 0])
    right_bound = np.amax([top_right[0], bottom_right[0], x2])
    y = int(right_bound - left_bound)
    # Similarly, determine upper and lower bounds to calculate height of stitched image
    upper_bound = np.amin([top_left[1], top_right[1], 0])
    lower_bound = np.amax([bottom_left[1], bottom_right[1], y2])
    x = int(lower_bound - upper_bound)
    # Delta values of x and y for translation matrix
    delta_x = -int(left_bound)
    delta_y = -int(upper_bound)
    # Define translation matrix to move upper left corner to (0,0)
    T = np.array([[1, 0, delta_x], [0, 1, delta_y], [0, 0, 1]])
    # Warp first image to perspective of second image
    # Swapped axes for correct result
    result = np.swapaxes(cv2.warpPerspective(np.swapaxes(image1, 0, 1), T @ H, (y, x)), 0, 1)
    # Find overlap region using delta values
    overlap = result[delta_x:x2+delta_x, delta_y:y2+delta_y]
    # Blending using average pixel value
    overlap = ((image2.astype(int)+overlap.astype(int))/2).astype(np.uint8)
    # Overlap image to create stitched image
    result[delta_x:x2+delta_x, delta_y:y2+delta_y] = overlap
    # cv2.imshow('r', result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return result


def stitch_img(image1, image2, H):
    # result_width = img1.shape[1] + img2.shape[1]
    # result_height = img1.shape[0] + img2.shape[0]
    # result = cv2.warpPerspective(img2, H, (result_width, result_height))
    # result[0:img1.shape[0], :img1.shape[1]] = img1
    # cv2.imshow('r', result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Find height and width of both images
    x1, y1 = tuple(image1.shape[:2])
    x2, y2 = tuple(image2.shape[:2])
    # Find estimated location of bounding box
    top_left = apply_homography(H, np.array([[0, 0]]))[0]
    top_right = apply_homography(H, np.array([[0, y1]]))[0]
    bottom_left = apply_homography(H, np.array([[x1, 0]]))[0]
    bottom_right = apply_homography(H, np.array([[x1, y1]]))[0]
    # top_left = cv2.warpPerspective(image2, H, (image1.shape[1]+image2.shape[1], image1.shape[0]+image2.shape[0]))
    # Determine left and right bounds to calculate width of stitched image
    left_bound = np.amin([top_left[0], bottom_left[0], 0])
    right_bound = np.amax([top_right[0], bottom_right[0], x2])
    y = int(right_bound - left_bound)
    # Similarly, determine upper and lower bounds to calculate height of stitched image
    upper_bound = np.amin([top_left[1], top_right[1], 0])
    lower_bound = np.amax([bottom_left[1], bottom_right[1], y2])
    x = int(lower_bound - upper_bound)
    # Delta values of x and y for translation matrix
    delta_x = -int(left_bound)
    delta_y = -int(upper_bound)
    # Define translation matrix to move upper left corner to (0,0)
    T = np.array([[1, 0, delta_x], [0, 1, delta_y], [0, 0, 1]])
    # Warp first image to perspective of second image
    # Swapped axes for correct result
    result = np.swapaxes(cv2.warpPerspective(np.swapaxes(image1, 0, 1), T @ H, (y, x)), 0, 1)
    # Find overlap region using delta values
    overlap = result[delta_x:x2+delta_x, delta_y:y2+delta_y]
    # Blending using average pixel value
    overlap = ((image2.astype(int)+overlap.astype(int))/2).astype(np.uint8)
    # Overlap image to create stitched image
    result[delta_x:x2+delta_x, delta_y:y2+delta_y] = overlap
    cv2.imshow('r', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return result

# Perform a really bad panorama stitching


def panorama_stiching(dir='../images/input'):
    # Load all images in given directory
    images = np.array([img.imread(dir+f) for f in os.listdir(dir)])
    # Start with first image
    result = images[0]
    for i in np.arange(1, images.shape[0]):
        features = np.array([np.zeros((3,))])
        # Analyze first image
        # Detect corners
        c_img = detect_corners(result)
        # Run ANMS
        corners = anms(c_img).astype(float)
        # Find feature descriptors
        desc = feature_descriptors(result, corners)
        # Store results
        features = np.append(features, [[1, corners, desc]], axis=0)
        # Analyze second image
        # Detect corners
        c_img = detect_corners(images[i])
        # Run ANMS
        corners = anms(c_img).astype(float)
        # Find feature descriptors
        desc = feature_descriptors(images[i], corners)
        # Store results again
        features = np.append(features, [[1, corners, desc]], axis=0)
        features = features[1:, 1:]
        # Match features
        matches = feature_matching(features)
        # If not matches found, fail algorithm
        if matches.shape[0] == 0:
            raise Exception("No matches found! Try using similar images")
        # Run RANSAC
        # H, inliers = ransac(matches, features)
        f1 = np.array(features[0, 0])
        f2 = np.array(features[1, 0])
        H, _ = cv2.findHomography(f1, f2, cv2.FM_RANSAC)
        # Stitch images to create new image
        result = stitch(result, images[i], H)
        # result = stitch_img(result, images[i], H)
    # Viola! Display a poorly sitched image
    # visualize_image(result)
    cv2.imshow('r', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def test(dir='../images/input'):
    # Load all images in given directory
    data = np.array([cv2.imread(dir+f) for f in os.listdir(dir)])
    # Start with first image
    img_0 = data[0]
    c_img_0 = detect_corners(img_0)
    # Run ANMS
    corners_0 = anms(c_img_0).astype(float)
    # Find feature descriptors
    desc0 = feature_descriptors(img_0, corners_0)
    img_1 = data[1]
    c_img_1 = detect_corners(img_1)
    # Run ANMS
    corners_1 = anms(c_img_1).astype(float)
    # Find feature descriptors
    desc1 = feature_descriptors(img_1, corners_1)
    # plt.figure(figsize=(15,15))
    # plt.subplot(121)
    # plt.axis('off')
    # plt.imshow(corners_0)
    # plt.subplot(122)
    # plt.axis('off')
    # plt.imshow(corners_1)
    # plt.show()
def main():
    # Add any Command Line arguments here
    # Parser = argparse.ArgumentParser()
    # Parser.add_argument('--NumFeatures', default=100, help='Number of best features to extract from each image, Default:100')

    # Args = Parser.parse_args()
    # NumFeatures = Args.NumFeatures
    """
    Read a set of images for Panorama stitching
    """
    file_dir = "../Data/Train/Set2/"
    # data = [i for i in os.listdir(file_dir) if i.endswith('.jpg')]
    """
	Corner Detection
	Save Corner detection output as corners.png
	"""

    """
	Perform ANMS: Adaptive Non-Maximal Suppression
	Save ANMS output as anms.png
	"""

    """
	Feature Descriptors
	Save Feature Descriptor output as FD.png
	"""

    """
	Feature Matching
	Save Feature Matching output as matching.png
	"""

    """
	Refine: RANSAC, Estimate Homography
	"""

    """
	Image Warping + Blending
	Save Panorama output as mypano.png
	"""
    # panorama_stiching(file_dir)
    test(file_dir)

if __name__ == '__main__':
    main()
