import numpy as np
from matplotlib import pyplot as plt
import sys,os
import cv2
from skimage.morphology import erosion, dilation, opening, closing, white_tophat
from skimage.morphology import disk
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


def binarize(img):
  mask = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,51,5)
  return(mask)


def distance(x1,y1,x2,y2,x0,y0):
  Dx = x2 - x1
  Dy = y2 - y1
  d = abs(Dy*x0 - Dx*y0 - x1*y2 +x2*y1)/((Dx**2 + Dy**2)**0.5)
  return(d)

img = cv2.imread(sys.argv[1],cv2.IMREAD_UNCHANGED)
img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

selem = disk(9)
eroded = erosion(gray, selem)



sigI,sigS = 7,7
kbsize = -1
blurred = cv2.bilateralFilter(eroded,kbsize,sigI,sigS)
blurred = cv2.bilateralFilter(blurred,kbsize,sigI,sigS)
blurred = cv2.bilateralFilter(blurred,kbsize,sigI,sigS)
blurred = cv2.GaussianBlur(blurred,(3,3),cv2.BORDER_DEFAULT)
edges = cv2.Canny(blurred,60,150)

# distance_thres = 900
c0 = edges.shape[0]/2
c1 = edges.shape[1]/2
lines = cv2.HoughLines(edges,1,np.pi/180,65)


scaler = StandardScaler()
X = lines.copy()
X = X.reshape(lines.shape[0],lines.shape[2])
scaler.fit(X)
X = scaler.transform(X)
clustering = DBSCAN(eps=0.1, min_samples=2).fit(X)
# lines = lines.reshape(lines.shape[0],-1,lines.shape[1])

labels = clustering.labels_
labToColour = {}

centriods = {}


for i,line in enumerate(lines):
  rho = line[0][0]
  theta = line[0][1]
  a = np.cos(theta)
  b = np.sin(theta)
  x0 = a*rho
  y0 = b*rho
  x1 = int(x0 + 1000*(-b))
  y1 = int(y0 + 1000*(a))
  x2 = int(x0 - 1000*(-b))
  y2 = int(y0 - 1000*(a))
  # dist = distance(x1,y1,x2,y2,c0,c1)
  # if dist<=distance_thres:
  if labels[i] == -1:
    continue
  if labels[i] not in labToColour:
    labToColour[labels[i]] = (int((np.random.rand(1)*255)[0]),int((np.random.rand(1)*255)[0]),int((np.random.rand(1)*255)[0]))
    centriods[labels[i]] = np.array([rho,theta,1])
  else:
    centriods[labels[i]] = centriods[labels[i]] + np.array([rho,theta,1])
  color = labToColour[labels[i]]
  # cv2.line(img,(x1,y1),(x2,y2),color,1)
  

for i in centriods:
  rho = centriods[i][0]/centriods[i][2]
  theta = centriods[i][1]/centriods[i][2]
  a = np.cos(theta)
  b = np.sin(theta)
  x0 = a*rho
  y0 = b*rho
  x1 = int(x0 + 1000*(-b))
  y1 = int(y0 + 1000*(a))
  x2 = int(x0 - 1000*(-b))
  y2 = int(y0 - 1000*(a))
  color = labToColour[i]
  cv2.line(img,(x1,y1),(x2,y2),color,3)

print(labToColour)





# cv2.imshow("Eroded",eroded)
cv2.imshow("Original",blurred)
cv2.imshow("Edges",edges)
cv2.imshow('houghlines',img)
# cv2.imshow("binary",binary)

cv2.waitKey(0)
cv2.destroyAllWindows()