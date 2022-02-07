import cv2
import numpy as np
from glob import glob 
import os
frame_num = 0
file_dir = "../Data/Train/Set3/"
# images = [cv2.resize(cv2.imread(file),(270, 480)) for file in os.listdir(file_dir) if file.endswith(".jpg")]

data = [i for i in os.listdir(file_dir) if i.endswith('.jpg')]
images = []
for i in range(len(data)):
    file_name = file_dir + data[i]
    img = cv2.imread(file_name)
    img = cv2.resize(img, (270, 480))
    images.append(images)
total_frame = len(images)
h = images[0].shape[0]
w = images[0].shape[1]
sift = cv2.xfeatures2d.SIFT_create()
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
result1 = np.zeros((h*3, w*3,3))
count  = np.zeros((h*3, w*3))
ones = np.ones(((h, w)))
while frame_num < total_frame:
    frame2 = images[frame_num]
    gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    kp2 = sift.detect(gray,None)
    dt2 = sift.compute(gray,kp2)[1]
    if frame_num == 0:   
        T      = np.eye(3)
        T[0,2] = result1.shape[1]-2*frame2.shape[1]
        T[1,2] = result1.shape[0]-2*frame2.shape[0]
        result = cv2.warpPerspective(frame2,T,(result1.shape[1],result1.shape[0])).astype(np.float)
        t_count= cv2.warpPerspective(ones,T,(result1.shape[1],result1.shape[0])).astype(np.float)
        count += t_count.astype(np.float)
        disp = result.copy()
        cv2.imshow('image',disp.astype(np.uint8))
        # cv2.imwrite("dataset2/result.jpg",disp.astype(np.uint8))
        kp1 = kp2
        dt1 = dt2
        frame1 = frame2
    
    matches = bf.match(dt2,dt1)
    #print('{}, # of matches:{}'.format(frame_num,len(matches)))

    matches = sorted(matches, key = lambda x:x.distance)    
    src = []
    dst = []
    for m in matches:
        src.append(kp2[m.queryIdx].pt + (1,))
        dst.append(kp1[m.trainIdx].pt + (1,))
            
    src = np.array(src,dtype=np.float)
    dst = np.array(dst,dtype=np.float)  
    M, mask = cv2.findHomography(src, dst, cv2.RANSAC)
    T = T.dot(M)
    warp_img = cv2.warpPerspective(frame2,T,(result1.shape[1],result1.shape[0])).astype(np.float)
    t_count  = cv2.warpPerspective(ones,T,(result1.shape[1],result1.shape[0])).astype(np.float)
    result1 += warp_img
    count += t_count.astype(np.float)
    t_count= count.copy()
    t_count[t_count == 0] = 1
    disp = result1.copy()
    disp[:,:,0] = result1[:,:,0] / t_count
    disp[:,:,1] = result1[:,:,1] / t_count
    disp[:,:,2] = result1[:,:,2] / t_count
    cv2.imshow('image',disp.astype(np.uint8))
    # cv2.imwrite("dataset2/result.jpg",disp.astype(np.uint8))
    kp1 = kp2
    dt1 = dt2
    frame1 = frame2
        
    key = cv2.waitKey(10) & 0xFF
    if key == 27:
        break
    frame_num += 1
    
cv2.waitKey()
cv2.destroyAllWindows()