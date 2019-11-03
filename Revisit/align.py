from matplotlib import pyplot as plt
import numpy as np
import sys,os
import cv2


sift = cv2.xfeatures2d.SIFT_create()
MIN_MATCH_COUNT = 1000
GOOD_MATCH_PERCENT = 0.10
bf = cv2.BFMatcher(cv2.NORM_L2 , crossCheck=True)



def getHomograpy(kp1,des1,gray2):
  kp2,des2 = sift.detectAndCompute(gray2,None)
  matches = bf.match(des2,des1)
  if len(matches) < MIN_MATCH_COUNT :
    return([False,None,None,None])
  matches.sort(key=lambda x: x.distance,reverse=False)
  print(len(matches))
  numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
  good = matches[:numGoodMatches]
  points1 = np.zeros((len(good), 2), dtype=np.float32)
  points2 = np.zeros((len(good), 2), dtype=np.float32)
  for i, match in enumerate(good):
    points1[i, :] = kp1[match.trainIdx].pt
    points2[i, :] = kp2[match.queryIdx].pt
  h, mask = cv2.findHomography(points1, points2, cv2.RANSAC,5.0)
  return([True,h,good,kp2])



dir = os.getcwd()
path = sys.argv[1]
folder = path.split("/")[1]
files = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
files.sort()


images = [cv2.imread(f,cv2.IMREAD_GRAYSCALE) for f in files ]

kp_temp,des_temp = sift.detectAndCompute(images[0],None)

for i in range(1,len(files)):
  l = getHomograpy(kp_temp,des_temp,images[i])
  if l[0]:
    images[i] = cv2.warpPerspective(images[i], l[1], (images[0].shape[1],images[0].shape[0])) 

try:
  os.mkdir(os.path.join(dir,"CorrectedAlignment/"+ folder))
except:
  print("Already exist")

for i in range(len(images)):
  cv2.imwrite(os.path.join(os.path.join(dir,"CorrectedAlignment/"+ folder),str(i+1)+".png"),images[i])


# cv2.imshow("Reference",images[0])
# cv2.imshow("Original",images[1])
# cv2.imshow("Aligned",im1Reg)
# cv2.waitKey(0)
# cv2.destroyAllWindows()