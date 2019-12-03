import sys
import cv2
# import drawCircles as dc
import numpy as np
from sympy import Point, Circle, Line, Ray

def check(c1,c2,r1,r2,line):
  d1 = abs(line[0]*c1[0] + line[1]*c1[1] + line[2])/((line[0]**2 + line[1]**2)**0.5) -r1 
  d2 = abs(line[0]*c2[0] + line[1]*c2[1] + line[2])/((line[0]**2 + line[1]**2)**0.5) -r2
  print(d1,d2)

def solve(r1,r2,v,origin):
  r = (r2 - r1)
  z = v[0]**2 + v[1]**2
  d = z - (r**2)
  if d < -1e-9:
    return(-1)
  d = (abs(d))**0.5
  a = (r*v[0] + v[1]*d)/(z)
  b = (r*v[1] - v[0]*d)/(z)
  c = r1 - (a*origin[0] + b*origin[1])
  return(a,b,c)

def getTangents(c1,c2,r1,r2):
  '''
    ci is the center of circlei
    ri is the radius of circlei
  '''
  tangents = []
  v = [c2[0]-c1[0],c2[1]-c1[1]] # v is the center is translated co-ordinate system
  for i in [-1,1]:
    for j in [-1,1]:
      line = solve(r1*i,r2*j,v,c1)
      # print(line)
      if line != -1:
        tangents.append(line)
  for line in tangents:
    check(c1,c2,r1,r2,line)
  return(tangents)


def drawtangents(temp,tangents,circles):
  # temp = np.zeros((605,641,3))
  for tan in [tangents[0],tangents[-1]]:
    cv2.line(temp,(-1000,int((-tan[0]*-1000  - tan[2])/tan[1] )),(1000,int((-tan[0]*1000  - tan[2])/tan[1])),(255,0,0),1)
  for cir in circles:
    cv2.circle(temp,(cir[0],cir[1]),cir[2],(0,0,255),1)
  cv2.imshow("temp",temp)
  cv2.waitKey(0)


# circles = dc.getCircles(sys.argv[1])
# print(circles)
# img = cv2.imread(sys.argv[1],cv2.IMREAD_UNCHANGED)

# circles  = [[165, 411, 44], [414, 408, 80]]
# for cir in circles:
#   cv2.circle(img,(cir[0],cir[1]),cir[2],(0,0,255),1)

# c1 = circles[0][:-1]
# c2 = circles[1][:-1]
# r1 = circles[0][2]
# r2 = circles[1][2]
# # cv2.imshow("image",img)
# tang = getTangents(c1,c2,r1,r2)
# drawtangents(img,tang,circles)
# cv2.destroyAllWindows()