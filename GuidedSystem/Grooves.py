import sys
import cv2
import tangents as tg
import numpy as np
from sympy import Point, Circle, Line, Ray
import enclosedArea as ea
import noiseReduction as nr
import process as pro


image = cv2.imread(sys.argv[1],cv2.IMREAD_UNCHANGED)

grayImg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
smooth = nr.rmSpecales(grayImg)
smooth = nr.smoothing(smooth)
smooth = nr.smoothing(smooth)


regions = (ea.getenclosedFigs(image,1))


# regions = [[(687, 239), (679, 389), (981, 430), (992, 265)],
#            [(527, 336), (499, 678), (657, 700), (686, 354)], 
#            [(631, 659), (578, 784), (994, 852), (966, 700)], 
#            [(973, 403), (903, 732), (1058, 782), (1101, 416)]]

# regions = [[(615, 280), (554, 713), (997, 785), (1058, 369)]]

print(regions)
newregs,medians = pro.getRegions(image,regions[0])
edges = cv2.Canny(smooth,50,75)
for AxisNum in range(4):
  xup,yup,xdwn,ydwn = pro.getdata(edges,medians[AxisNum],newregs[AxisNum])
  pro.fitline(image,medians[AxisNum],xup,yup,(255,0,0))
  pro.fitline(image,medians[AxisNum],xdwn,ydwn,(0,0,255))

# exp,mask = crop_region(regions[0],image)
# exp = nr.rmSpecales(cv2.cvtColor(exp, cv2.COLOR_BGR2GRAY))
# edges = cv2.Canny(exp,25,50)


cv2.imshow("image",image)
# cv2.imshow("enclosed region",edges*(mask//255))
# cv2.imshow("encl",exp)

# cv2.imshow("smooth",smooth)
cv2.waitKey(0)
cv2.destroyAllWindows()




