import cv2
import numpy as np
import sys

drawing = False # true if mouse is pressed
ix,iy = -1,-1
curentNoRect = 0
img,imgcopy = -1,-1
rect = []


def draw_rectangle(event,x,y,flags,param):
  global ix,iy,drawing,img,imgcopy,curentNoRect
  if event == cv2.EVENT_LBUTTONDOWN:
    drawing = True
    ix,iy = x,y
  elif event == cv2.EVENT_MOUSEMOVE:
    if drawing == True:
      img = np.copy(imgcopy)
      cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),1)
  elif event == cv2.EVENT_LBUTTONUP:
    drawing = False
    img = np.copy(imgcopy)
    imgcopy = img
    curentNoRect += 1
    rect.append([ix,iy,x,y])
    cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),1)


def get_rectangles(im):
  global img,imgcopy
  # img = np.zeros((512,512,3), np.uint8)
  img = cv2.imread(im,cv2.IMREAD_UNCHANGED)
  imgcopy = np.copy(img)
  cv2.namedWindow('image')
  cv2.setMouseCallback('image',draw_rectangle)
  while(1):
    cv2.imshow('image',img)
    k = cv2.waitKey(10) & 0xFF
    if k == 27 or curentNoRect == 4:
      break
  cv2.destroyAllWindows()

get_rectangles(sys.argv[1])