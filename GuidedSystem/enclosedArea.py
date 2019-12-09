import cv2
import numpy as np
import sys

drawing = False 
ix,iy = -1,-1
img,imgcopy = -1,-1
closure = []
lineMade = False
selectedPoints = 0

def enclosefig(event,x,y,flags,param):
  global ix,iy,drawing,img,imgcopy,closure,selectedPoints
  if event == cv2.EVENT_LBUTTONDOWN:
    drawing = True
    ix,iy = x,y
    selectedPoints += 1
    closure.append((ix,iy))
    if selectedPoints > 1:
      cv2.line(img,closure[-2],closure[-1],(0,255,0),1)
    imgcopy = img
    img = np.copy(imgcopy)
    if selectedPoints == 4:
      drawing = False
      cv2.line(img,closure[0],closure[-1],(0,255,0),1)
  elif event == cv2.EVENT_MOUSEMOVE:
    if drawing == True:
      img = np.copy(imgcopy)
      cv2.line(img,closure[-1],(x,y),(0,255,0),1)
  



def getenclosedFigs(image,NoOfFigs):
  global closure,img,imgcopy,selectedPoints
  # img = np.zeros((512,512,3), np.uint8)
  img = np.copy(image)
  imgcopy = np.copy(img)
  enclosures = []
  for i in range(NoOfFigs):
    closure = []
    cv2.namedWindow('image')
    cv2.setMouseCallback('image',enclosefig)
    while(1):
      cv2.imshow('image',img)
      k = cv2.waitKey(10) & 0xFF
      if k == 27 or selectedPoints==4:
        enclosures.append(closure)
        closure = []
        selectedPoints = 0
        break
    cv2.destroyAllWindows()
  cv2.imshow("result",img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  return(enclosures)




# print(getenclosedFigs(image,4))