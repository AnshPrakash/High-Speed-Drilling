import cv2
import sys
import numpy as np
import Grooves
import Triangles
import drawCircles as dr
import enclosedArea as ea
import process as pro

def PolygonArea(corners):
  n = len(corners) # of corners
  area = 0.0
  for i in range(n):
    j = (i + 1) % n
    area += corners[i][0] * corners[j][1]
    area -= corners[j][0] * corners[i][1]
  area = abs(area)/2.0
  return area


def rotate(pt,angle):
  # theta in degrees
  theta = np.radians(angle)
  c, s = np.cos(theta), np.sin(theta)
  R = np.array([[c,-s],[s,c]])
  return(R.dot(pt))


def scoreRadii(data):
  # print("Variance",np.var(data))
  # return(np.exp(-np.var(data)))
  return(1 - (max(data)-min(data))/(max(data)) )


def scoreCircles(circles):
  cords = [cir[:-1]  for cir in circles]
  radii = [cir[-1] for cir in circles] 
  (tplfx, tplfy), (width, height), angle = rect = cv2.minAreaRect(np.array(cords))
  angles = list(range(0,180,1))
  fit_area = min([PolygonArea(np.array([rotate(cords[i],angle) for i in range(len(cords))]))  for angle in angles])
  areaScore = ((fit_area/(width*height)))
  radiiScore = scoreRadii(radii)
  # print("areaScore",areaScore)
  # print("Radii Score",scoreRadii(radii))
  return((areaScore + radiiScore)/2 )


def scoreGrooves(GrooveLines,corners):
  grooveEqForm = []
  for lines in GrooveLines:
    [x0,y0],[x1,y1] = lines[0]
    A = -(y1 - y0)
    B = (x1 - x0)
    C1 = -y0*B -x0*A
    [x0,y0],[x1,y1] = lines[1]
    C2 = -y0*B -x0*A
    grooveEqForm.append([A,B,C1,C2])
  widths = [pro.distancebwLines(A,B,C1,C2) for A,B,C1,C2 in grooveEqForm]
  widthScore = (1 - (max(widths)-min(widths))/(max(widths)))
  (tplfx, tplfy), (width, height), angle = rect = cv2.minAreaRect(np.array(corners))
  angles = list(range(0,180,1))
  fit_area = min([PolygonArea(np.array([rotate(corners[i],angle) for i in range(len(corners))]))  for angle in angles])
  areaScore = ((fit_area/(width*height)))
  return((areaScore + widthScore)/2 )


  

def scoreTri():
  pass





'''
  image1 is just the template with circular holes
  image2 only include the rectangular grooves
  image3 is  include the diagonal grooves
'''

images =  [cv2.imread(sys.argv[i+1],cv2.IMREAD_UNCHANGED) for i in range(3)]

# # Score  Circles
# img = np.copy(images[0])
# circles = dr.getCircles(img)
# # print("Circle Score",scoreCircles(circles))

# get Templates
templates = [[],[]]
templates[0] = (ea.getenclosedFigs(images[1],1,(0,0,0)))[0]
templates[1] = (ea.getenclosedFigs(images[2],1,(0,0,0)))[0]
M12 = cv2.getPerspectiveTransform(np.float32(templates[0]),np.float32(templates[1]))


# Score Rectangular groves
img = np.copy(images[1])
GrooveLines,corners = Grooves.GetGrooveInfo(img)
Tribases = pro.getbaseFromGrooveLines(GrooveLines,M12)
# print(GrooveLines)
print("Groove Score",scoreGrooves(GrooveLines,corners))


# Score Diagonal grooves
img = np.copy(images[2])
TriLines = Triangles.GetTriangles(img,Tribases)

print(TriLines)

