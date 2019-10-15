import numpy as np
import pywt
from matplotlib import pyplot as plt
import sys
import cv2




# x = pywt.data.camera().astype(np.float32)
x = cv2.imread(sys.argv[1],cv2.IMREAD_GRAYSCALE)
# x = x/255.0
# delta = 0.01
# x = x/np.exp(np.mean(np.log(delta + x)))
kernel = np.ones((9,9))
gradient = cv2.morphologyEx(x, cv2.MORPH_GRADIENT, kernel)
# tophat = cv2.morphologyEx(x, cv2.MORPH_TOPHAT, kernel)
gradient = gradient.astype('uint8')
edges = cv2.Canny(gradient,60,150)

cv2.imshow("Original",x)
cv2.imshow("gradient",gradient)
cv2.imshow("Edges",edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

