import numpy as np
import cv2
import process as pro
import drawCircles as dr

def createmask(circles,shape):
  Y, X = np.ogrid[:shape[0], :shape[1]]
  finalmask = np.zeros((shape[0],shape[1]))
  # finalmask = finalmask.astype('bool')
  probdist = 20
  for circle in circles:
    # dist_from_center = np.sqrt((X - 100)**2 + (Y- 100)**2)
    dist_from_center = np.sqrt((X - circle[0])**2 + (Y- circle[1])**2)
    mask = (dist_from_center <= circle[2] + probdist)
    # print(mask.dtype)
    finalmask = finalmask + 2*mask #this is a boolean OR operation
    mask = (dist_from_center <= circle[2])
    finalmask = finalmask - 1*mask
    # print(circles[0][2])
    # print(dist_from_center)
    # print(dist_from_center.dtype)
    # print("____________")
  # print(finalmask.dtype)
  finalmask = np.uint8(finalmask)
  cv2.imshow("mask",finalmask*100)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  return(finalmask)

def getcontour(image,mask):
  img = np.copy(image)
  bgdModel = np.zeros((1,65),np.float64)
  fgdModel = np.zeros((1,65),np.float64)
  cv2.grabCut(img,mask,None,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_MASK)
  print("set",np.unique(mask))
  # from matplotlib import pyplot as plt
  # plt.imshow(mask),plt.colorbar(),plt.show()
  mask = np.where((mask==2)|(mask==0),0,1).astype('uint8')
  # print("set",np.unique(edged))
  _, contours, hierarchy  = cv2.findContours(mask*255,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
  # _, contours, hierarchy  = cv2.findContours(mask*255,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
  # areas = [cv2.contourArea(c) for c in contours]
  # sorted_areas = np.sort(areas)
  contours = sorted(contours,key = lambda c : cv2.contourArea(c),reverse = True)
  contours = contours[:4] # only 4 circle required other countour is noise
  image = np.copy(img)
  cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
  colors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0)]
  ellipses = []
  circles = []
  #fit ellipse 
  for cnt,color in zip(contours,colors):
    ellipse = cv2.fitEllipse(cnt)
    (x,y),radius = cv2.minEnclosingCircle(cnt)
    circles.append([int(x),int(y),int(radius)])
    ellipses.append(ellipse)
    cv2.ellipse(img,ellipse,color,2)
    cv2.circle(img,(int(x),int(y)),int(radius),(253,155,125),2)
  circles = pro.clockify(circles)
  # print("circles",circles)
  # print("Ellipses",ellipses)
  cv2.imshow("contour",image)
  # cv2.imshow("mask",mask*255)
  cv2.imshow("ellipsesCircles",img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  return(ellipses,circles)


# image = cv2.imread(sys.argv[1],cv2.IMREAD_UNCHANGED)
# image = cv2.cvtColor(image,cv2.COLOR_BGRA2BGR)
# # image = cv2.resize(image,(1700,1000))
# img = np.copy(image)
# circles = dr.getCircles(image)
# # circles =  [[637, 220, 46], [1069, 292, 59], [1015, 702, 50], [605, 659, 51]]
# print(circles)

# mask = createmask(circles,img.shape)
# getcontour(img,mask)


