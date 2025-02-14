import cv2
import sys
import numpy as np
import Grooves
import Triangles
import drawCircles as dr
import enclosedArea as ea
import noiseReduction as nr
import process as pro
import fitCountour as ftc
import adjustLine as adjL

def PolygonArea(corners):
  n = len(corners) # of corners
  area = 0.0
  for i in range(n):
    j = (i + 1) % n
    area += corners[i][0] * corners[j][1]
    area -= corners[j][0] * corners[i][1]
  area = abs(area)/2.0
  return area


def rotate(pt,angle):
  # theta in degrees
  theta = np.radians(angle)
  c, s = np.cos(theta), np.sin(theta)
  R = np.array([[c,-s],[s,c]])
  return(R.dot(pt))


def scoreRadii(data):
  # print("Variance",np.var(data))
  # return(np.exp(-np.var(data)))
  return(1 - (max(data)-min(data))/(max(data)) )


def scoreCircles(circles):
  cords = [cir[:-1]  for cir in circles]
  radii = [cir[-1] for cir in circles] 
  (tplfx, tplfy), (width, height), angle = rect = cv2.minAreaRect(np.array(cords))
  angles = list(range(0,180,1))
  fit_area = min([PolygonArea(np.array([rotate(cords[i],angle) for i in range(len(cords))]))  for angle in angles])
  areaScore = ((fit_area/(width*height)))
  radiiScore = scoreRadii(radii)
  # print("areaScore",areaScore)
  # print("Radii Score",scoreRadii(radii))
  return((areaScore + radiiScore)/2 )


def scoreGrooves(GrooveLines,corners):
  grooveEqForm = []
  for lines in GrooveLines:
    [x0,y0],[x1,y1] = lines[0]
    A = -(y1 - y0)
    B = (x1 - x0)
    C1 = -y0*B -x0*A
    [x0,y0],[x1,y1] = lines[1]
    C2 = -y0*B -x0*A
    grooveEqForm.append([A,B,C1,C2])
  widths = [pro.distancebwLines(A,B,C1,C2) for A,B,C1,C2 in grooveEqForm]
  widthScore = (1 - (max(widths)-min(widths))/(max(widths)))
  (tplfx, tplfy), (width, height), angle = rect = cv2.minAreaRect(np.array(corners))
  angles = list(range(0,180,1))
  fit_area = min([PolygonArea(np.array([rotate(corners[i],angle) for i in range(len(corners))]))  for angle in angles])
  areaScore = ((fit_area/(width*height)))
  return((areaScore + widthScore)/2)



def scoreTri(TriLines,img):
  TriEq = [] # each list contains 3 tuple of size 3 representing A,B,C 
  TriPoints = [] # each list contains 3 points representing the vertices of triangle
  TriArea = []
  TriLengths = []
  for triangle in TriLines:
    TriEq.append([pro.getEquation(line) for line in triangle])
  for triangle in TriEq:
    intersectionPts = []
    for i in range(2):
      for j in range(i+1,3):
        intersectionPts.append(pro.intersection(triangle[i],triangle[j]))
    TriPoints.append(intersectionPts)
  colors = [(0,0,255),(0,255,0),(255,0,0),(0,0,0)]
  for tri,color in zip(TriPoints,colors):
    for pt in tri:
      cv2.circle(img,(int(pt[0]),int(pt[1])),4,color,-1)
    TriArea.append(PolygonArea(tri))
    lengths = []
    for i in range(2):
      for j in range(i+1,3):
        lengths.append(pro.distancebwPoints(tri[i],tri[j]))
    TriLengths.append(lengths)
  TriAreaScore = 1 - ((np.max(TriArea) - np.min(TriArea))/np.max(TriArea))
  TriLengths = np.array(TriLengths)
  TriLengthScores = 1 - ((np.max(TriLengths,axis=0) - np.min(TriLengths,axis=0))/np.max(TriLengths,axis=0)) #for congrugency 
  FinalScore = (TriAreaScore + np.mean(TriLengthScores))/2.0
  cv2.imshow("intersectionPoints",img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  return(FinalScore)


def scoreAlignment(corners1,corners2):
  c1 = np.mean(corners1,axis=0)
  c2 = np.mean(corners2,axis=0)
  diagonal = (np.array(corners1[0]) - np.array(corners1[2]))
  diagonal = (diagonal[0]**2 + diagonal[1]**2)**0.5
  scoreAlignment = (((c1-c2)[0]**2 + (c1-c2)[1]**2 )**0.5)/diagonal
  return(scoreAlignment)

def scoreEllipses(ellipses):
  avgE = 0
  for ellipse in ellipses:
    a,b = max(ellipse[1]),min(ellipse[1])
    e = (1 - (b/a)**2)**(0.5)
    avgE = avgE + e
  avgE =avgE/len(ellipses)
  return(1 - avgE)

def scoreAdjustedGroove(adjGrooves):
  score = 0
  for groove in GrooveLines: 
    l1 = np.array(groove[0][1]) - np.array(groove[0][0])
    l2 = np.array(groove[1][1]) - np.array(groove[1][0])
    l1 = l1/((l1[0]**2 + l1[1]**2)**0.5)
    l2 = l2/((l2[0]**2 + l2[1]**2)**0.5)
    cos = np.dot(l1,l2)
    score = score + cos
  score = score/4
  return(score)




'''
  image1 is just the template with circular holes
  image2 only include the rectangular grooves
  image3 is  include the diagonal grooves
'''

images =  [cv2.imread(sys.argv[i+1],cv2.IMREAD_UNCHANGED) for i in range(3)]


# Score  Circles
img = np.copy(images[0])
circles = dr.getCircles(img)
mask = ftc.createmask(circles,img.shape)
img = cv2.cvtColor(images[0],cv2.COLOR_BGRA2BGR)
ellipses,circles= ftc.getcontour(img,mask)
print("Circle Score",scoreCircles(circles))
print("Ellipse Score",scoreEllipses(ellipses))



# get Templates
templates = [[],[]]
templates[0] = (ea.getenclosedFigs(images[1],1,(0,0,0)))[0]
templates[1] = (ea.getenclosedFigs(images[2],1,(0,0,0)))[0]
# templates = [[(543, 175), (1145, 266), (1059, 877), (450, 773)], [(297, 163), (903, 157), (947, 774), (358, 768)]]

M12 = cv2.getPerspectiveTransform(np.float32(templates[0]),np.float32(templates[1]))


# Score Rectangular groves
img = np.copy(images[1])
grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
smooth = nr.rmSpecales(grayImg)
smooth = nr.smoothing(smooth)
smooth = nr.smoothing(smooth)
edges = cv2.Canny(smooth,50,75)
GrooveLines,corners = Grooves.GetGrooveInfo(img,edges)
# GrooveLines = [[[[604, 626], [642, 380]], [[533, 615], [571, 369]]], [[[703, 344], [976, 392]], [[716, 268], [989, 316]]], [[[1026, 452], [978, 710]], [[1111, 467], [1063, 725]]], [[[912, 752], [650, 692]], [[894, 831], [632, 771]]]]
# corners = [(620, 293), (1074, 372), (994, 802), (557, 703)]

AlignmentScore = scoreAlignment(corners,templates[0]) # whether the groove are drawn in template
Tribases = pro.getbaseFromGrooveLines(GrooveLines,M12)

adjustedGrooveLines = []
img = np.copy(images[1])
for groove in GrooveLines: 
  l1 = adjL.adjustLine(groove[0],edges,img)
  l2 = adjL.adjustLine(groove[1],edges,img)
  adjustedGrooveLines.append([l1,l2])

# adjustedGrooveLines = [[[[488, 1373], [800, -601]], [[414, 1362], [726, -613]]], [[[-284, 167], [1684, 513]], [[-267, 67], [1695, 447]]], [[[830, 1441], [1210, -521]], [[926, 1458], [1306, -504]]], [[[912, 752], [650, 692]], [[894, 831], [632, 771]]]]

for groove in adjustedGrooveLines:
  l1 = groove[0]
  l2 = groove[1]
  cv2.line(img,(l1[0][0],l1[0][1]),(l1[1][0],l1[1][1]),(255,255,0),1) 
  cv2.line(img,(l2[0][0],l2[0][1]),(l2[1][0],l2[1][1]),(255,255,0),1) 
cv2.imshow("Adjusted Lines",img)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(adjustedGrooveLines)
print("Groove Score",scoreGrooves(GrooveLines,corners))
print("Adjusted Groove Score",scoreAdjustedGroove(adjustedGrooveLines))

# Score Diagonal grooves
img = np.copy(images[2])
grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
smooth = nr.rmSpecales(grayImg)
smooth = nr.rmSpecales(grayImg)
smooth = nr.smoothing(smooth)
edges = cv2.Canny(smooth,50,75)
TriLines = Triangles.GetTriangles(img,Tribases,edges)

# TriLines =        [
#                     [[[597, 471], [456, 362]], [[605, 493], [481, 604]],np.array([[461, 602],[442, 342]], dtype=np.int32)], 
#                     [[[674, 411], [787, 309]], [[637, 419], [496, 310]], np.array([[491, 296],[756, 303]], dtype=np.int32)], 
#                     [[[725, 500], [848, 619]], [[724, 466], [837, 364]], np.array([[818, 355],[836, 620]], dtype=np.int32)], 
#                     [[[650, 544], [526, 655]], [[685, 541], [808, 660]], np.array([[785, 673],[522, 662]], dtype=np.int32)]
#                   ]

print("Triangle Score",scoreTri(TriLines,img))

