import cv2
import numpy as np
import process as pro
import time

drawing = False # true if mouse is pressed
curentNoCircles = 0
cx,cy = -1,-1
img,imgcopy = -1,-1
circles = []

# mouse callback function
def draw_circle(event,x,y,flags,param):
  global drawing,mode,cx,cy,img,imgcopy,curentNoCircles
  if event == cv2.EVENT_LBUTTONDOWN:
    drawing = True
    cx,cy = x,y
  elif event == cv2.EVENT_MOUSEMOVE:
    if drawing == True:
      img = np.copy(imgcopy)
      radius = int(((x-cx)**2 + (y - cy)**2)**0.5)
      cv2.circle(img,(cx,cy),radius,(0,0,255),1)
  elif event == cv2.EVENT_LBUTTONUP:
    drawing = False
    img = np.copy(imgcopy)
    imgcopy = img
    curentNoCircles += 1
    radius = int(((x-cx)**2 + (y - cy)**2)**0.5)
    circles.append([cx,cy,radius])
    cv2.circle(img,(cx,cy),radius,(0,0,255),1)



def getCircles(imag):
  global img,imgcopy,circles,curentNoCircles
  # img = np.zeros((512,512,3), np.uint8)
  # img = cv2.imread(im,cv2.IMREAD_UNCHANGED)
  img = imag
  imgcopy = np.copy(img)
  cv2.namedWindow('image')
  cv2.setMouseCallback('image',draw_circle)

  while(1):
    cv2.imshow('image',img)
    k = cv2.waitKey(10) & 0xFF
    if k == 27 or curentNoCircles == 4:
      break
  # print(circles)
  # cv2.destroyAllWindows()
  cv2.imshow("circles",img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  circles =  pro.clockify(circles)
  return(circles)
