import sys
import cv2
import numpy as np
import drawTriangles as tr
import noiseReduction as nr
import process as pro


image = cv2.imread(sys.argv[1],cv2.IMREAD_UNCHANGED)
grayImg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
smooth = nr.rmSpecales(grayImg)
smooth = nr.smoothing(smooth)
smooth = nr.smoothing(smooth)

regions = (tr.getenclosedFigs(np.copy(image),1))
# regions = [[(832, 557), (637, 350), (1057, 454), (979, 892), (547, 741)]]
newregs,medians = pro.getRegionsTri(image,regions[0])
edges = cv2.Canny(smooth,50,75)
# print("median",len(medians))
# print(medians)
for AxisNum in range(len(medians)):
  xup,yup,xdwn,ydwn = pro.getdata(edges,medians[AxisNum],newregs[AxisNum])
  pro.fitline(image,medians[AxisNum],xup,yup,(255,0,0))
  pro.fitline(image,medians[AxisNum],xdwn,ydwn,(0,0,255))

cv2.imshow("results?",image)
cv2.waitKey(0)
cv2.destroyAllWindows()


