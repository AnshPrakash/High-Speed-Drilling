import sys
import cv2
import drawCircles as dc
import tangents as tg
import numpy as np
from sympy import Point, Circle, Line, Ray



circles = dc.getCircles(sys.argv[1])
# print(circles)
img = cv2.imread(sys.argv[1],cv2.IMREAD_UNCHANGED)

# circles  = [[165, 411, 44], [414, 408, 80]]
# for cir in circles:
#   cv2.circle(img,(cir[0],cir[1]),cir[2],(0,0,255),1)

c1 = circles[0][:-1]
c2 = circles[1][:-1]
r1 = circles[0][2]
r2 = circles[1][2]
# cv2.imshow("image",img)
tang = tg.getTangents(c1,c2,r1,r2)
tg.drawtangents(img,tang,circles)
cv2.destroyAllWindows()