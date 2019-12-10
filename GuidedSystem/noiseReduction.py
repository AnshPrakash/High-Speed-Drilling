import cv2
import numpy as np
from skimage.morphology import erosion, dilation, opening, closing, white_tophat
from skimage.morphology import black_tophat, skeletonize, convex_hull_image
from skimage.morphology import disk

def rmSpecales(img):
  '''
    img should be greyscale
  '''
  selem = disk(4)
  exp = opening(img, selem)
  return(exp)

def smoothing(img):
  '''
    img should be greyscale
  '''
  sigI,sigS = 9,9
  # ksize = 31
  kbsize = -1
  smooth = cv2.bilateralFilter(img,kbsize,sigI,sigS)
  return(smooth)
