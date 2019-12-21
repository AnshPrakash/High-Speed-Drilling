import sys
import cv2
import numpy as np
import drawTriangles as tr
import noiseReduction as nr
import process as pro


def GetTriangles(img,bases):
  # image = cv2.imread(sys.argv[1],cv2.IMREAD_UNCHANGED)
  image = np.copy(img)
  grayImg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  smooth = nr.rmSpecales(grayImg)
  smooth = nr.rmSpecales(grayImg)
  smooth = nr.smoothing(smooth)
  # smooth = nr.smoothing(smooth)
  cv2.imshow("smooth",smooth)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  regions = (tr.getenclosedFigs(np.copy(image),1))
  # regions = [[(832, 557), (637, 350), (1057, 454), (979, 892), (547, 741)]]
  newregs,medians = pro.getRegionsTri(image,regions[0])
  edges = cv2.Canny(smooth,50,75)
  # print("median",len(medians))
  # print(medians)
  sides = []
  # bases = []
  for AxisNum in range(len(medians) - 4):
    xup,yup,xdwn,ydwn = pro.getdata(edges,medians[AxisNum],newregs[AxisNum])
    l1 = pro.fitline(image,medians[AxisNum],xup,yup,(255,0,0))
    l2 = pro.fitline(image,medians[AxisNum],xdwn,ydwn,(0,0,255))
    sides.append([l1,l2])
  # for AxisNum in range(4,len(medians)):
  #   xuni,yuni = pro.getUniData(edges,medians[AxisNum],newregs[AxisNum])
  #   baseline = pro.fitline(image,medians[AxisNum],xuni,yuni,(0,255,0))
  #   bases.append(baseline)

  TriLines = []
  for ti in range(4):
    TriLines.append([sides[ti][1],sides[(ti-1)%4][0],bases[ti]])
  cv2.imshow("results",image)
  cv2.waitKey(0)
  colors = [(0,0,255),(255,0,0),(0,255,0),(0,255,255)] 
  image = np.copy(img)
  for i in range(4):
    for j in range(3):
      cv2.line(image,tuple(TriLines[i][j][0]),tuple(TriLines[i][j][1]),colors[i],3)
  cv2.imshow("Triangles",image)  
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  return(TriLines)

# print(GetTriangles(cv2.imread(sys.argv[1],cv2.IMREAD_UNCHANGED)))

