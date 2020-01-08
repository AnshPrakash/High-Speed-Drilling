import numpy as np
import cv2
import sys
import process as pro
import Grooves
import math

def pointForm(rho,theta):
  a = np.cos(theta)
  b = np.sin(theta)
  x0 = a*rho
  y0 = b*rho
  x1 = int(x0 + 1000*(-b))
  y1 = int(y0 + 1000*(a))
  x2 = int(x0 - 1000*(-b))
  y2 = int(y0 - 1000*(a))
  return([[x1,y1],[x2,y2]])


def drawlines(edges,img,lines,origin):
  for line in lines:
    for rho,theta in line:
      [[x1,y1],[x2,y2]] = pointForm(rho,theta)
      cv2.line(edges,(x1,y1),(x2,y2 ),(0,0,255),2)
      cv2.line(img,(x1+origin[0],y1 + origin[1]),(x2 + origin[0],y2 + origin[1]),(0,0,255),2)
  cv2.imshow("Edges",edges)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

def normalForm(line):
  A,B,C = pro.getEquation(line)
  rho = np.abs(pro.distance(A,B,C,0,0))
  theta = math.atan(B/(A+0.0000001))
  if theta<0:
    theta = math.pi + theta
  return(rho,theta)

def getMinDist(lines,refl):
  dist = [ abs(refl[1]-line[1])  for line in lines]
  value = min(dist)
  minIdx = dist.index(value)
  return(minIdx,value)

def fitLine(houghlines,orgline,origin,edges):
  refLine = [ [pt[0] - origin[0],pt[1] - origin[1]] for pt in orgline]
  refNormline = normalForm(refLine)
  cv2.circle(edges, (refLine[0][0],refLine[0][1]), 3, (25,255,0), 3)
  cv2.circle(edges, (refLine[1][0],refLine[1][1]), 3, (25,255,0), 3)
  normLines = [ [line[0][0]/refNormline[0],line[0][1]/math.pi] for line in houghlines]
  ref = [1,refNormline[1]/math.pi]
  bestlineIdx,deviation = getMinDist(normLines,ref)
  bestline = houghlines[bestlineIdx]
  if deviation > math.radians(20):
    retline = orgline
    deviation = 0 
  else:
    retline = pointForm(bestline[0][0],bestline[0][1])
  retline = [[retline[0][0] + origin[0],retline[0][1] +origin[1]],[retline[1][0] + origin[0],retline[1][1] + origin[1]]]
  cv2.imshow("ref",edges)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  return(retline,deviation)


def getbestLine(croppedEdges,region,line,img):
  lines = cv2.HoughLines(croppedEdges,1,np.pi/180,40)
  reg = pro.clockify(region)
  edges = cv2.merge([croppedEdges,croppedEdges,croppedEdges])
  box = cv2.boundingRect(np.array(reg))
  origin = box[0],box[1]
  print((reg[0][0],reg[0][1]))
  cv2.circle(img, (reg[0][0],reg[0][1]), 3, (255,0,0), 3)
  if lines is None:
    return(line,0)
  # drawlines(edges,img,lines,origin)
  bestline,deviation = fitLine(lines,line,origin,edges)
  return(bestline,1 - deviation)

def adjustLine(line,edges,image):
  img = np.copy(image)
  uplmt = 30
  x0,y0 = line[0]
  x1,y1 = line[1]
  A = -(y1 - y0)
  B = (x1 - x0)
  C = -y0*B -x0*A
  u1 = complex(line[0][0],line[0][1])
  u2 = complex(line[1][0],line[1][1])
  v = u2 - u1
  length = np.absolute(v)
  v = v/length
  vt = complex(v.imag,-v.real)
  rect = [u1 -vt*uplmt,u1  + vt*uplmt, u2 +vt*uplmt, u2 - vt*uplmt]
  region = [[int(rec.real),int(rec.imag)] for rec in rect]
  cropped,mask = pro.crop_region(region,edges)
  for j in range(3):
    cv2.line(img,(int(rect[j].real),int(rect[j].imag)),(int(rect[j+1].real),int(rect[j+1].imag)),(0,255,0),1)
  cv2.line(img,(int(rect[-1].real),int(rect[-1].imag)),(int(rect[0].real),int(rect[0].imag)),(0,255,0),1) 
  cv2.circle(img,(int(u1.real),int(u1.imag)),5,(0,0,255),-1)
  cv2.circle(img,(int(u2.real),int(u2.imag)),5,(0,0,255),-1)
  newLine,_ = getbestLine(cropped*(mask//255),region,line,img)
  cv2.line(img,(newLine[0][0],newLine[0][1]),(newLine[1][0],newLine[1][1]),(255,255,0),1) 
  cv2.imshow("adjust Line",img)
  # cv2.imshow("rect",cropped*(mask//255))
  # cv2.imshow("mask",mask)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  return(newLine)


import noiseReduction as nr
img = cv2.imread(sys.argv[1],cv2.IMREAD_UNCHANGED)
groovelines = [[[[601, 638], [640, 380]], [[530, 627], [569, 369]]], [[[700, 342], [972, 387]], [[712, 267], [984, 312]]], [[[1024, 445], [982, 706]], [[1110, 458], [1068, 719]]], [[[914, 751], [645, 703]], [[899, 835], [630, 787]]]]
grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
smooth = nr.rmSpecales(grayImg)
smooth = nr.smoothing(smooth)
smooth = nr.smoothing(smooth)
edges = cv2.Canny(smooth,50,75)
# print(groovelines[0])
adjustLine(groovelines[3][0],edges,img)