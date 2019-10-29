from helper import *


def binarize(img):
  mask = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
  return(mask)

def BilateralBlur(img):
  sigI,sigS = 5,5
  kbsize = -1
  blurred = cv2.bilateralFilter(img,kbsize,sigI,sigS)
  blurred = cv2.bilateralFilter(blurred,kbsize,sigI,sigS)
  blurred = cv2.bilateralFilter(blurred,kbsize,sigI,sigS)
  return(blurred)


def RadialGaussianFilter(size,sigma,r0,f0):
  x0 = size/2
  y0 = size/2
  y,x = np.meshgrid(list(range(size)),list(range(size)))
  r = ((x - x0) **2 + (y - y0)**2)**0.5
  complx = np.exp((2*3.14*f0*(r - r0))*1.0j)
  ker = np.exp(-3.14*((r - r0)**2)/(sigma**2))*complx
  return(ker/(np.abs(np.sum(ker))))

def getGfilters(f0,sigma,r0):
  # s = [1,1.3,1.5,1.8,1.9]
  ker = RadialGaussianFilter(300,sigma,r0,f0)
  cv2.imshow("Filter"+str(r0),(ker.real -np.amin(ker.real))/(np.amax(ker.real) - np.amin(ker.real)))  
  # cv2.waitKey(0)
  return(ker.real)

def CircularGabor(img,f0,sigma,r0):
  filters = getGfilters(f0,sigma,r0)
  fimg = cv2.filter2D(img/255.0, cv2.CV_64F, filters)
  return(fimg)

img = cv2.imread(sys.argv[1],cv2.IMREAD_GRAYSCALE)
corners  = [[0,0],[0,img.shape[1]-1],[img.shape[0]-1,img.shape[1]-1] , [img.shape[0]-1,0]] 
# corners  = [[25,43],[0,img.shape[1]-1],[img.shape[0]-1,img.shape[1]-1] , [img.shape[0]-1,0]] 

size = int(max(img.shape[0],img.shape[1])/3.1)

croppedImgs = [0]*4
croppedImgs[0] = img[corners[0][0]:corners[0][0] + size, corners[0][1]:corners[0][1]+size] #Top Left
croppedImgs[1] = img[corners[1][0]:corners[1][0] + size, corners[1][1] - size:corners[1][1]] #Top Right
croppedImgs[2] = img[corners[2][0] - size:corners[2][0], corners[2][1] - size:corners[2][1]] # Bottom right
croppedImgs[3] = img[corners[3][0] - size:corners[3][0], corners[3][1]:corners[3][1] + size] # Bottom Left

lowFimgs = []
for limg in croppedImgs:
  lowFimgs.append(BilateralBlur(limg))


binImgs = []
for bimg in lowFimgs:
  binImgs.append(binarize(bimg))

# cv2.imshow("Binaries",)
# cv2.waitKey(0)


idx = 0
posRadius = np.array([50,55,60,70])
f0 = 1/10
sigma = 100
fimg = CircularGabor(1 - binImgs[idx]/255,f0,sigma,posRadius[0])
for r0 in posRadius[1:]:
  np.maximum(CircularGabor(1 - binImgs[idx]/255,f0,sigma,r0),fimg,fimg)


# bright = CircularGabor(fimg,0,10,3)

# center1 = (np.unravel_index(bright.argmax(), bright.shape))
center = (np.unravel_index(fimg.argmax(), fimg.shape))
print(center)
print(lowFimgs[idx].shape)
cv2.circle(lowFimgs[idx],(center[1],center[0]),2,(23,42,255),3)
# cv2.circle(lowFimgs[idx],(center1[1],center1[0]),2,(13,42,55),3)

# print(fimg[center[0]][center[1]])
# print(fimg[center[1]][center[0]])
# print(np.amax(fimg))
cv2.imshow("Original",255 -binImgs[idx])
cv2.imshow("Cropped",lowFimgs[idx])

cv2.imshow("Filtered",(fimg - np.amin(fimg))/(np.amax(fimg) - np.amin(fimg)))  
# cv2.imshow("Brightest",(bright - np.amin(bright))/(np.amax(bright) - np.amin(bright)))  

cv2.waitKey(0)
cv2.destroyAllWindows()

# responses = []
# for limg in lowFimgs:
#   responses.append(CircularGabor(limg))


# cv2.imshow("Original",img)

# for i,img in enumerate(lowFimgs):
#   cv2.imshow("Blurred"+str(i),img)

# for i,res in enumerate(responses):
#   cv2.imshow("Response"+str(i),res)


# cv2.waitKey(0)
# cv2.destroyAllWindows()
