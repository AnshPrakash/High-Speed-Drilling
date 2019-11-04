import numpy as np
from matplotlib import pyplot as plt
import sys,os
import cv2
from skimage.morphology import erosion, dilation, opening, closing, white_tophat
from skimage.morphology import disk
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sympy import Point,Line


def binarize(img):
  mask = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,51,5)
  return(mask)


def getPoints(rho,theta):
  a = np.cos(theta)
  b = np.sin(theta)
  x0 = a*rho
  y0 = b*rho
  x1 = int(x0 + 1000*(-b))
  y1 = int(y0 + 1000*(a))
  x2 = int(x0 - 1000*(-b))
  y2 = int(y0 - 1000*(a))
  return(Point(x1,y1),Point(x2,y2))


def getAllAngles(lines):
  '''
    returns a list of angles bw all the lines nC2
  '''
  angles = []
  for i in range(len(lines)):
    for j in range(i+1,len(lines)):
      p1,p2 = getPoints(lines[i][0],lines[i][1]) 
      p11,p22 = getPoints(lines[j][0],lines[j][1]) 
      l1 = Line(p1,p2)
      l2 = Line(p11,p22)
      ang = l1.angle_between(l2).evalf()
      angles.append(min(ang,np.pi - ang))
  return(np.array(angles))




def getSquares(lines):
  '''
    eps is the acceptable variance for considering it to be a square
  '''
  squares = []
  minn = 100000
  for i in range(len(lines)):
    for j in range(i+1,len(lines)):
      for k in range(j+1,len(lines)):
        for l in range(k+1,len(lines)):
          psq = [lines[i],lines[j],lines[k],lines[l]]
          angles = getAllAngles(psq)
          print(abs(np.mean(angles) - np.pi/2))
          if abs(np.mean(angles) - np.pi/2) < minn:
            squares = psq
            minn = abs(np.mean(angles) - np.pi/2)
  print("minn",minn)
  return(squares)


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
lines = cv2.HoughLines(edges,1,np.pi/180,65)


scaler = StandardScaler()
X = lines.copy()
X = X.reshape(lines.shape[0],lines.shape[2])
scaler.fit(X)
X = scaler.transform(X)
clustering = DBSCAN(eps=0.3, min_samples=2).fit(X)
# lines = lines.reshape(lines.shape[0],-1,lines.shape[1])

labels = clustering.labels_
labToColour = {}
centriods = {}

shwSquare = img.copy() # remove this later

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


centriodLines = []

for i in centriods:
  rho = centriods[i][0]/centriods[i][2]
  theta = centriods[i][1]/centriods[i][2]
  centriodLines.append([rho,theta])
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

squares = centriodLines
if len(centriodLines)>4:
  squares = getSquares(centriodLines)

print(squares)

for i,_ in enumerate(squares):
  rho = squares[i][0]
  theta = squares[i][1]
  a = np.cos(theta)
  b = np.sin(theta)
  x0 = a*rho
  y0 = b*rho
  x1 = int(x0 + 1000*(-b))
  y1 = int(y0 + 1000*(a))
  x2 = int(x0 - 1000*(-b))
  y2 = int(y0 - 1000*(a))
  color = (int((np.random.rand(1)*255)[0]),int((np.random.rand(1)*255)[0]),int((np.random.rand(1)*255)[0]))
  cv2.line(shwSquare,(x1,y1),(x2,y2),color,2)





# cv2.imshow("Eroded",eroded)
cv2.imshow("Original",blurred)
cv2.imshow("Edges",edges)
cv2.imshow('houghlines',img)
cv2.imshow('Square',shwSquare)
# cv2.imshow("binary",binary)

cv2.waitKey(0)
cv2.destroyAllWindows()