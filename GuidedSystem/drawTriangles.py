import numpy as np
import process as pro
import cv2
# import sys

drawing = False 
ix,iy = -1,-1
img,imgcopy = -1,-1
closure = []
selectedPoints = 0

def enclosetriangle(event,x,y,flags,param):
  global ix,iy,drawing,img,imgcopy,closure,selectedPoints
  if event == cv2.EVENT_LBUTTONDOWN:
    # print("x,y",x,y)
    drawing = True
    ix,iy = x,y
    selectedPoints += 1
    closure.append((ix,iy))
    if selectedPoints > 1:
      cv2.line(img,closure[0],closure[-1],(0,255,0),3)
    imgcopy = img
    img = np.copy(imgcopy)
    if selectedPoints == 5:
      drawing = False
      # print("before",closure[1:])
      closure[1:] = pro.clockify(closure[1:])
      for i in range(1,4):
        cv2.line(img,closure[i],closure[i+1],(0,255,0),3)
      cv2.line(img,closure[-1],closure[1],(0,255,0),3)
  elif event == cv2.EVENT_MOUSEMOVE:
    if drawing == True:
      img = np.copy(imgcopy)
      cv2.line(img,closure[0],(x,y),(0,255,0),3)
  

def getenclosedFigs(image,NoOfFigs):
  global closure,img,imgcopy,selectedPoints
  # img = np.zeros((512,512,3), np.uint8)
  img = np.copy(image)
  imgcopy = np.copy(img)
  enclosures = []
  for i in range(NoOfFigs):
    closure = []
    cv2.namedWindow('image')
    cv2.setMouseCallback('image',enclosetriangle)
    while(1):
      cv2.imshow('image',img)
      k = cv2.waitKey(10) & 0xFF
      if k == 27 or selectedPoints==5:
        enclosures.append(closure)
        closure = []
        selectedPoints = 0
        break
    cv2.destroyAllWindows()
  cv2.imshow("result",img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  return(enclosures)

# image = cv2.imread(sys.argv[1],cv2.IMREAD_UNCHANGED)
# regions = (getenclosedFigs(image,1))
# print(regions[0])
