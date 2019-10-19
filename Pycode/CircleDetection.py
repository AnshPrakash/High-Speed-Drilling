from helper import *

img = cv2.imread(sys.argv[1],cv2.IMREAD_GRAYSCALE)
mask = cv2.imread(sys.argv[2],cv2.IMREAD_GRAYSCALE)
img = img/255.0
kernel = np.ones((5,5), np.uint8) 
# mask = cv2.dilate(mask, kernel, iterations=1) 
mask = mask/255.0

blurred = anisotropicFilter(img,1,3,31)
blurred = blurred*255
blurred = blurred.astype('uint8')
sigI,sigS = 9,9
# ksize = 31
kbsize = -1

blurred = cv2.bilateralFilter(blurred,kbsize,sigI,sigS)
blurred = cv2.bilateralFilter(blurred,kbsize,sigI,sigS)
blurred = cv2.bilateralFilter(blurred,kbsize,sigI,sigS)
blurred = blurred/255.0
kernel = np.ones((9,9))
# gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
# gradient = gradient*255
# gradient = gradient.astype('uint8')
# edges = cv2.Canny(gradient,30,100)

filters = build_filters()
response = processGabor(blurred,filters)
response = response*255
response = response.astype('uint8')
edges = cv2.Canny(response,30,100)
edges = edges/255.0
res = edges*(1-mask)
EdgeMask = edges + mask


############ 
# res = res*255
# res = res.astype('uint8')

# circles = cv2.HoughCircles(res,cv2.HOUGH_GRADIENT,1,20,
#                             param1=50,param2=30,minRadius=0,maxRadius=0)
# circles = np.uint16(np.around(circles))
# for i in circles[0,:]:
#   # draw the outer circle
#   cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),2)
#   # draw the center of the circle
#   cv2.circle(img,(i[0],i[1]),2,(0,0,255),3)

##########
# response = processGabor(img,filters)
# print(Gauss2d(5,1,3,20.0))


cv2.imshow("Original",img)
cv2.imshow("Anisotropic Filtering",blurred)
cv2.imshow("Edges",edges)
cv2.imshow("Gabor Filter",response)
cv2.imshow("Mask",mask)
cv2.imshow("MaskEdge",EdgeMask)
cv2.imshow("ResEdge",res)




cv2.waitKey(0)
cv2.destroyAllWindows()
