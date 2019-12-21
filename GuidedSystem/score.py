import cv2
import sys
import numpy as np
import Grooves
import Triangles
import drawCircles as dr

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
  print(GrooveLines)
  print(corners)

def scoreTri():
  pass

'''
  image1 is just the template with circular holes
  image2 only include the rectangular grooves
  image3 is  include the diagonal grooves
'''

images =  [cv2.imread(sys.argv[i+1],cv2.IMREAD_UNCHANGED) for i in range(3)]

# Score  Circles
img = np.copy(images[0])
circles = dr.getCircles(img)
# circles = [[678, 386, 40], [1051, 448, 44], [966, 845, 40], [589, 775, 45]]
# circles = [[678, 386, 80], [1051, 448, 44], [966, 845, 40], [589, 775, 45]]
# print(circles)
# print("Circle Score",scoreCircles(circles))


# Score Rectangular groves
img = np.copy(images[1])
GrooveLines,corners = Grooves.GetGrooveInfo(img)
print("Groove Score",scoreGrooves(GrooveLines,corners))
# print(GrooveLines)
# print("_______________")
# print(corners)

# Score Diagonal grooves
img = np.copy(images[2])
TriLines = Triangles.GetTriangles(img)

