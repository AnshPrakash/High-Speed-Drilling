from helper import *


def binarize(img):
  mask = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
  return(mask)

def BilateralBlur(img):
  sigI,sigS = 5,5
  kbsize = -1
  blurred = cv2.bilateralFilter(img,kbsize,sigI,sigS)
  # blurred = cv2.bilateralFilter(blurred,kbsize,sigI,sigS)
  # blurred = cv2.bilateralFilter(blurred,kbsize,sigI,sigS)
  return(blurred)

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

kernel = np.ones((3,3),np.uint8)
for i in range(len(binImgs)):
  binImgs[i] = cv2.morphologyEx(binImgs[i], cv2.MORPH_OPEN, kernel,iterations = 1)

minRadius = int((size/3)/2)
maxRadius = int((size*(1.414)))
minDist = size
resolution = 1
CannyHigh = 80
AccumThresh = 50
for j,res in enumerate(binImgs):
  circles = cv2.HoughCircles(res,cv2.HOUGH_GRADIENT,resolution,minDist,
                              param1=CannyHigh,param2=AccumThresh,minRadius=minRadius,maxRadius=maxRadius)
  if circles is None:
    continue
  circles = np.uint16(np.around(circles))
  # print(circles[0][0])
  lowFimgs[j] = cv2.cvtColor(lowFimgs[j],cv2.COLOR_GRAY2BGR)
  for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(lowFimgs[j],(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv2.circle(lowFimgs[j],(i[0],i[1]),2,(0,0,255),3)



cv2.imshow("Original",img)

for i,img in enumerate(binImgs):
  cv2.imshow("Adaptive Theresholding"+str(i),img)


for i,img in enumerate(lowFimgs):
  cv2.imshow("Blurred"+str(i),img)



cv2.waitKey(0)
cv2.destroyAllWindows()
