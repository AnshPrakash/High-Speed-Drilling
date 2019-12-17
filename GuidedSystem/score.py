import cv2
import sys
import numpy as np
import Grooves
import Triangles
import drawCircles as dr


'''
  image1 is just the template with circular holes
  image2 only include the rectangular grooves
  image3 is  include the diagonal grooves
'''
images =  [ cv2.imread(sys.argv[i+1],cv2.IMREAD_UNCHANGED) for i in range(3)]

# Score  Circles
img = np.copy(images[0])
circles = dr.getCircles(img)
# print(circles)

# Score Rectangular groves
img = np.copy(images[1])
GrooveLines,corners = Grooves.GetGrooveInfo(img)
# print(GrooveLines)
# print("_______________")
# print(corners)

# Score Diagonal grooves
img = np.copy(images[2])
TriLines = Triangles.GetTriangles(img)

