import numpy as np
import pywt
from matplotlib import pyplot as plt
import sys,os
import cv2

THERSHOLD = 0.75

def findReflection(img):
  mask = np.zeros_like(img,dtype=np.uint8)
  mask[img>THERSHOLD] = 1
  return(mask)


def gaussian_pyramid(img, num_levels):
  lower = img.copy()
  gaussian_pyr = [lower]
  for i in range(num_levels):
    lower = cv2.pyrDown(lower)
    gaussian_pyr.append(np.float64(lower))
  return gaussian_pyr
 

def laplacian_pyramid(gaussian_pyr):
  laplacian_top = gaussian_pyr[-1]
  num_levels = len(gaussian_pyr) - 1  
  laplacian_pyr = [laplacian_top]
  for i in range(num_levels,0,-1):
    size = (gaussian_pyr[i - 1].shape[1], gaussian_pyr[i - 1].shape[0])
    gaussian_expanded = cv2.pyrUp(gaussian_pyr[i], dstsize=size)
    laplacian = np.subtract(gaussian_pyr[i-1], gaussian_expanded)
    laplacian_pyr.append(laplacian)
  return laplacian_pyr


# Now blend the two images wrt. the mask
def blend(laplacian_A,laplacian_B,mask_pyr):
    LS = []
    for la,lb,mask in zip(laplacian_A,laplacian_B,mask_pyr):
        ls = lb * mask + la * (1.0 - mask)
        LS.append(ls)
    return LS

def reconstruct(laplacian_pyr):
  laplacian_top = laplacian_pyr[0]
  laplacian_lst = [laplacian_top]
  num_levels = len(laplacian_pyr) - 1
  for i in range(num_levels):
    size = (laplacian_pyr[i + 1].shape[1], laplacian_pyr[i + 1].shape[0])
    laplacian_expanded = cv2.pyrUp(laplacian_top, dstsize=size)
    laplacian_top = cv2.add(laplacian_pyr[i+1], laplacian_expanded)
    laplacian_lst.append(laplacian_top)
  return laplacian_lst

def Lapblend(img1,img2,mask,num_levels = 6):
  gaussian_pyr_1 = gaussian_pyramid(img1, num_levels)
  laplacian_pyr_1 = laplacian_pyramid(gaussian_pyr_1)
  # For image-2, calculate Gaussian and Laplacian
  gaussian_pyr_2 = gaussian_pyramid(img2, num_levels)
  laplacian_pyr_2 = laplacian_pyramid(gaussian_pyr_2)
  # Calculate the Gaussian pyramid for the mask image and reverse it.
  mask_pyr_final = gaussian_pyramid(mask, num_levels)
  mask_pyr_final.reverse()
  # Blend the images
  add_laplace = blend(laplacian_pyr_1,laplacian_pyr_2,mask_pyr_final)
  # Reconstruct the images
  final  = reconstruct(add_laplace)
  return(final[num_levels])


def correctReflection(img,mask):
  # dst = LaplacianBlend(img,)
  img = img
  # img = img.astype(np.uint8)
  # mask = mask.astype(np.uint8)
  img2 = np.zeros_like(mask)
  img2[mask == 1.0] = np.sum(img[mask!=1.0])/(img.size - np.sum(mask) + 0.01)
  dst = Lapblend(img,img2,mask)
  # dst = cv2.inpaint(img,mask,9,cv2.INPAINT_TELEA)
  # dst = cv2.inpaint(img,mask,3,cv2.INPAINT_NS)
  # dst = img2
  return(dst)




files = [f for f in os.listdir(sys.argv[1]) if os.path.isfile(os.path.join(sys.argv[1], f))]
for f in files:
# img = cv2.imread(sys.argv[1],cv2.IMREAD_GRAYSCALE)
  img = cv2.imread(os.path.join(sys.argv[1],f),cv2.IMREAD_GRAYSCALE)
  img = img/255.0
  mask = findReflection(img)
  detected = np.copy(img)
  detected[mask==1] = 0.0
  ref = correctReflection(img,mask)
  cv2.imwrite("ReflectionCorrected/"+ f,ref*255)
  cv2.imwrite("Masks/"+ f,mask*255)





# cv2.imshow("Original",img)
# cv2.imshow("Reflection Detected",detected)
# cv2.imshow("Reflection Corrected",ref)



cv2.waitKey(0)
cv2.destroyAllWindows()
