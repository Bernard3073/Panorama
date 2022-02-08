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
           so the matrix has dimension Nx2 (N>=4).
    - dest: (x,y) coordinates of N destination pixels, where coordinates are row vectors,
            so the matrix has dimension Nx2 (N>=4).
    Returns:
    - the homography matrix such that H @ [x_src, y_src, 1].T = [x_dest, y_dest, 1].T
    Author:
    - Yu Fang
    """
    N = src.shape[0]
    if N != dest.shape[0]:
        raise ValueError("src and diff should have the same dimension")
    src_h = np.hstack((src, np.ones((N, 1))))
    A = np.array([np.block([[src_h[n], np.zeros(3), -dest[n, 0] * src_h[n]],
                            [np.zeros(3), src_h[n], -dest[n, 1] * src_h[n]]])
                  for n in range(N)]).reshape(2 * N, 9)
    [_, _, V] = np.linalg.svd(A)
    # take the right singular vector x corresponding to the least singular value
    # s.t. ||Ax - 0||^2 is minimized
    return V.T[:, 8].reshape(3, 3)


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
    return cv2.cornerHarris(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 5, 5, 0.04)
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # gray = np.float32(gray)
    # dst = cv2.cornerHarris(image, 3, 3, 0.04)
    # dst = cv2.dilate(dst, None)
    # image[dst > 0.01*dst.max()] = [255, 0, 0]
    # # cv2.imshow('dst', image)
    # # cv2.waitKey(0)
    # # cv2.destroyAllWindows()
    # return dst

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
    kp, desc = sift.compute(image, keypoints)
    # desc = cv2.GaussianBlur(desc, (5, 5), 0)
    # desc = cv2.resize(desc, (8, 8), interpolation=cv2.INTER_AREA)
    # desc = np.reshape(desc, (64,))

    # img = cv2.drawKeypoints(image, kp, image)
    # cv2.imshow('f', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # return desc, kp
    # # Normalize feature descriptors
    # return (desc - np.mean(desc, axis=0))/np.std(desc, axis=0)
    return desc


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

# def feature_matching(desc1, desc2):
#     # FLANN parameters
#     FLANN_INDEX_KDTREE = 0
#     index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
#     search_params = dict(checks=50)  # or pass empty dictionary

#     flann = cv2.FlannBasedMatcher(index_params, search_params)

#     matches = flann.knnMatch(desc1, desc2, k=2)

#     # Apply ratio test
#     good = []
#     for m, n in matches:
#         if m.distance < 0.8 * n.distance:
#             good.append(m)
#     return good

# Refine matches and estimate robust homography with RANSAC
def ransac(matches, features):
    n_max = 10000
    H_best, inliers_best, n_best = None, None, -np.inf
    for n in np.arange(n_max):
        # Select 4 random pairs of matches
        rand = matches[random.sample(range(0, matches.shape[0]), 4)]
        p1 = features[0, 0][rand[:,0]]
        p2 = features[1, 0][rand[:,1]]
        # Estimate homography using 4 random pairs of matches
        H = est_homography(p1, p2)
        # Estimate points with homography
        H_p1 = apply_homography(H, features[0,0][matches[:,0]])
        actual = features[1,0][matches[:,1]]
        # Compare estimated points with actual points
        dist = np.linalg.norm(H_p1 - actual, axis=1)
        # Ensure difference is around 3 or 4 pixels
        inliers = matches[dist < 4]
        # Keep the largest set of inliers and estimated homography
        if inliers.shape[0] > n_best:
            n_best = inliers.shape[0]
            inliers_best = inliers
            H_best = H
        # Terminate is 90% of keypoints have been reached
        if n_best >= 0.9*H_p1.shape[0]:
            break
    return H_best, inliers_best



# def ransac(f1, f2):
#     N = np.inf
#     sample_count = 0
#     p = 0.99
#     threshold = 0.5
#     max_num_inlier = 0
#     while N > sample_count:
#         # Select 4 random pairs of matches
#         rand = np.random.choice(len(f1), size=4)
#         rand_p1 = np.array([f1[i] for i in rand])
#         rand_p2 = np.array([f2[i] for i in rand])
#         # Estimate homography using 4 random pairs of matches
#         H = computeH_norm(rand_p1, rand_p2)
#         inlier_count = 0
#         inliers = []
#         for p1, p2 in zip(f1, f2):
#             p2_homo = np.append(p2, 1).reshape(3, 1)
#             # Estimate points with homography
#             p1_est = H @ p2_homo
#             p1_est = (p1_est/p1_est[2])[:2].reshape(1, 2)
#             if cv2.norm(p1 - p1_est) <= threshold:
#                 inlier_count += 1
#                 inliers.append(1)
#             else:
#                 inliers.append(0)

#         if inlier_count > max_num_inlier:
#             max_num_inlier = inlier_count
#             H_best = H
#             inliers.append(p1)
#         inlier_ratio = inlier_count / len(f1)
#         if np.log(1 - (inlier_ratio**8)) == 0:
#             continue
#         N = np.log(1-p) / np.log(1 - (inlier_ratio**8))
#         sample_count += 1

#     return H_best, inliers

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
    result = np.swapaxes(cv2.warpPerspective(
        np.swapaxes(image1, 0, 1), T @ H, (y, x)), 0, 1)
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
    result_width = image1.shape[1] + image2.shape[1]
    result_height = image1.shape[0] + image2.shape[0]
    result = cv2.warpPerspective(image2, H, (result_width, result_height))
    result[0:image1.shape[0], :image1.shape[1]] = image1
    cv2.imshow('d', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # # Find height and width of both images
    # x1, y1 = tuple(image1.shape[:2])
    # x2, y2 = tuple(image2.shape[:2])
    # # Find estimated location of bounding box
    # top_left = apply_homography(H, np.array([[0, 0]]))[0]
    # top_right = apply_homography(H, np.array([[0, y1]]))[0]
    # bottom_left = apply_homography(H, np.array([[x1, 0]]))[0]
    # bottom_right = apply_homography(H, np.array([[x1, y1]]))[0]
    # # top_left = cv2.warpPerspective(image2, H, (image1.shape[1]+image2.shape[1], image1.shape[0]+image2.shape[0]))
    # # Determine left and right bounds to calculate width of stitched image
    # left_bound = np.amin([top_left[0], bottom_left[0], 0])
    # right_bound = np.amax([top_right[0], bottom_right[0], x2])
    # y = int(right_bound - left_bound)
    # # Similarly, determine upper and lower bounds to calculate height of stitched image
    # upper_bound = np.amin([top_left[1], top_right[1], 0])
    # lower_bound = np.amax([bottom_left[1], bottom_right[1], y2])
    # x = int(lower_bound - upper_bound)
    # # Delta values of x and y for translation matrix
    # delta_x = -int(left_bound)
    # delta_y = -int(upper_bound)
    # # Define translation matrix to move upper left corner to (0,0)
    # T = np.array([[1, 0, delta_x], [0, 1, delta_y], [0, 0, 1]])
    # # Warp first image to perspective of second image
    # # Swapped axes for correct result
    # result = np.swapaxes(cv2.warpPerspective(np.swapaxes(image1, 0, 1), T @ H, (y, x)), 0, 1)
    # # Find overlap region using delta values
    # overlap = result[delta_x:x2+delta_x, delta_y:y2+delta_y]
    # # Blending using average pixel value
    # overlap = ((image2.astype(int)+overlap.astype(int))/2).astype(np.uint8)
    # # Overlap image to create stitched image
    # result[delta_x:x2+delta_x, delta_y:y2+delta_y] = overlap
    # cv2.imshow('r1', result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return result

# Perform a really bad panorama stitching
def panorama_stiching(dir='../images/input'):
    # Load all images in given directory
    images = np.array([img.imread(dir+f) for f in sorted(os.listdir(dir))])
    for i in range(images.shape[0]-1):
        features = np.array([np.zeros((3,))])
        # Analyze first image
        
        # Detect corners
        c_img = detect_corners(images[i])
        # Run ANMS
        corners = anms(c_img).astype(float)
        # Find feature descriptors
        desc = feature_descriptors(images[i], corners)
        # Store results
        features = np.append(features, [[1, corners, desc]], axis=0)
        # Analyze second image
        # Detect corners
        c_img = detect_corners(images[i+1])
        # Run ANMS
        corners = anms(c_img).astype(float)
        # Find feature descriptors
        desc = feature_descriptors(images[i+1], corners)
        # Store results again
        features = np.append(features, [[1, corners, desc]], axis=0)
        features = features[1:,1:]
        # Match features
        matches = feature_matching(features)
        # If not matches found, fail algorithm
        if matches.shape[0] == 0:
            raise Exception("No matches found! Try using similar images")
        # Run RANSAC
        H, _ = ransac(matches, features)
        # Stitch images to create new image
        result = stitch(images[i], images[i+1], H)
        # result = stitch_img(result, images[i], H)
        # visualize_image(result)

        # result = stitch_img(result, images[i], H)
    # Viola! Display a poorly sitched image
    visualize_image(result)

def main():
    # Add any Command Line arguments here
    # Parser = argparse.ArgumentParser()
    # Parser.add_argument('--NumFeatures', default=100, help='Number of best features to extract from each image, Default:100')

    # Args = Parser.parse_args()
    # NumFeatures = Args.NumFeatures
    """
    Read a set of images for Panorama stitching
    """
    file_dir = "../Data/Train/Set3/"
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
    panorama_stiching(file_dir)


if __name__ == '__main__':
    main()
