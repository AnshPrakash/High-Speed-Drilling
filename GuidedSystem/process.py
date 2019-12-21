import cv2
import numpy as np


'''
  points are in closewise order everywhere
'''

def crop_region(reg,img):
  pts = np.array([[l[0],l[1]] for l in reg])
  # pts = np.array([[int(l.real),int(l.imag)] for l in reg])
  rect = cv2.boundingRect(pts)
  x,y,w,h = rect
  croped = img[y:y+h, x:x+w].copy()
  pts = pts - pts.min(axis=0)
  mask = np.zeros(croped.shape[:2], np.uint8)
  cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
  dst = cv2.bitwise_and(croped, croped, mask=mask)
  bg = np.ones_like(croped, np.uint8)*255
  cv2.bitwise_not(bg,bg, mask=mask)
  dst2 = bg + dst
  # cv2.imwrite("garbage/croped.png", croped)
  # cv2.imwrite("garbage/mask.png", mask)
  # cv2.imwrite("garbage/dst.png", dst)
  # cv2.imwrite("garbage/dst2.png", dst2)
  return(croped,mask)


def distance(A,B,C,x,y):
  d = (A*x + B*y + C)/((A**2 + B**2)**0.5)
  return(d)

def distancebwLines(A,B,C1,C2):
  return(abs(C2 - C1)/((A*A + B*B)**0.5))

def sepAroundLine(line,xs,ys):
  # print(line)
  x0,y0 = line[0]
  x1,y1 = line[1]
  A = -(y1 - y0)
  B = (x1 - x0)
  C = -y0*B -x0*A
  xup,yup = [],[]
  xdwn,ydwn = [],[]
  for x,y in zip(xs,ys):
    if (distance(A,B,C,x,y)< 0 ):
      xdwn.append(x)
      ydwn.append(y)
    else:
      xup.append(x)
      yup.append(y)
  return(xup,yup,xdwn,ydwn)
      

  
def getdata(edges,line,reg):
  '''
    line have two point representing that line
  '''
  temp = cv2.merge([edges,edges,edges])
  rect = cv2.boundingRect(np.array(reg))
  origin = rect[0],rect[1]
  cropped,mask = crop_region(reg,edges)
  y,x = np.where(cropped*(mask//255) == 255 )
  x = x + origin[0]
  y = y + origin[1] 
  xup,yup,xdwn,ydwn = sepAroundLine(line,x,y)
  for i,j in zip(xup,yup):
    cv2.circle(temp,(i,j),1,(255,0,0),-1)
  for i,j in zip(xdwn,ydwn):
    cv2.circle(temp,(i,j),1,(0,0,255),-1)
  cv2.imshow("edge",temp)
  # cv2.imshow("mask",cropped*(mask//255))
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  return(xup,yup,xdwn,ydwn)

def getUniData(edges,line,reg):
  '''
    line have two point representing that line
  '''
  temp = cv2.merge([edges,edges,edges])
  rect = cv2.boundingRect(np.array(reg))
  origin = rect[0],rect[1]
  cropped,mask = crop_region(reg,edges)
  y,x = np.where(cropped*(mask//255) == 255 )
  x = x + origin[0]
  y = y + origin[1]
  for i,j in zip(x,y):
    cv2.circle(temp,(i,j),1,(0,0,255),-1)
  cv2.imshow("edge",temp)
  # cv2.imshow("mask",cropped*(mask//255))
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  return(x,y)

  

def getRegions(image,cords):
  '''
    returns the region of interest for finding the groove lines 
    and the corresponding medial axial given by user
  '''
  img = np.copy(image)
  newregs = []
  lines = []
  # l = [0,1,2,3,0]
  l = [3,0,1,2,3]
  for i in range(len(l)-1):
    reg = [cords[l[i]],cords[l[i+1]]]
    u1 = complex(reg[0][0], reg[0][1])
    u2 = complex(reg[1][0], reg[1][1])
    cv2.circle(img,(int(u1.real),int(u1.imag)),4,(255,5,0),-1)
    cv2.circle(img,(int(u2.real),int(u2.imag)),4,(255,5,0),-1)
    v = (u2 - u1)
    length = np.absolute(v)
    v = v/length
    vt = complex(v.imag,-v.real)
    d = length*0.2
    # uplmt = 80
    uplmt = 70
    u1 = u1 + v*d
    u2 = u2 - v*d
    rect = [u1 -vt*uplmt,u1  + vt*uplmt, u2 +vt*uplmt, u2 - vt*uplmt ]
    for j in range(3):
      cv2.line(img,(int(rect[j].real),int(rect[j].imag)),(int(rect[j+1].real),int(rect[j+1].imag)),(0,255,0),1)
    cv2.line(img,(int(rect[-1].real),int(rect[-1].imag)),(int(rect[0].real),int(rect[0].imag)),(0,255,0),1) 
    cv2.circle(img,(int(u1.real),int(u1.imag)),5,(0,0,255),-1)
    cv2.circle(img,(int(u2.real),int(u2.imag)),5,(0,0,255),-1)
    rect = [[int(rec.real),int(rec.imag)] for rec in rect]
    lines.append([[int(u1.real),int(u1.imag)],[int(u2.real),int(u2.imag)]])
    newregs.append(rect)
  cv2.imshow("getRegions",img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  # print("LINES",lines[1])
  # print("******")
  return((newregs,lines))



def fitline(img,line,xs,ys,color):
  x0,y0 = line[0]
  x1,y1 = line[1]
  A = -(y1 - y0)
  B = (x1 - x0)
  C = -y0*B -x0*A
  u0 = complex(line[0][0],line[0][1])
  u1 = complex(line[1][0],line[1][1])
  fun = lambda x, y : (distance(A,B,C,x,y))
  dists = fun(np.array(xs),np.array(ys))
  estimate = np.median(dists) # replace it with anything
  v = u1 - u0
  length = np.absolute(v)
  v = v/length
  vt = complex(v.imag,-v.real)
  u0 = u0 - vt*estimate
  u1 = u1 - vt*estimate
  # color = (0,255,0)
  res = [[int(u0.real),int(u0.imag)],[int(u1.real),int(u1.imag)]]
  cv2.line(img,tuple(res[0]),tuple(res[1]),color,4)
  return(res)


def clockify(cords):
  pts = cords.copy()
  pts = sorted(pts,key = (lambda x: x[1]))
  pts[:2] = sorted(pts[:2],key = (lambda x: x[0]))
  pts[2:] = sorted(pts[2:],key = (lambda x: -x[0]))
  return(pts)

def getRegionsTri(image,cords):
  '''
    returns the region of interest for finding the groove lines 
    and the corresponding medial axial given by user
  '''
  img = np.copy(image)
  newregs = []
  lines = []
  l = [0,1,0,2,0,3,0,4,4,1,1,2,2,3,3,4] 
  for i in range(0,len(l),2):
    reg = [cords[l[i]],cords[l[i+1]]]
    u1 = complex(reg[0][0], reg[0][1])
    u2 = complex(reg[1][0], reg[1][1])
    cv2.circle(img,(int(u1.real),int(u1.imag)),4,(255,5,0),-1)
    cv2.circle(img,(int(u2.real),int(u2.imag)),4,(255,5,0),-1)
    v = (u2 - u1)
    length = np.absolute(v)
    v = v/length
    vt = complex(v.imag,-v.real)
    d = length*0.2
    # uplmt = 80
    uplmt = 70
    u1 = u1 + v*d
    u2 = u2 - v*d
    rect = [u1 -vt*uplmt,u1  + vt*uplmt, u2 +vt*uplmt, u2 - vt*uplmt ]
    for j in range(3):
      cv2.line(img,(int(rect[j].real),int(rect[j].imag)),(int(rect[j+1].real),int(rect[j+1].imag)),(0,255,0),1)
    cv2.line(img,(int(rect[-1].real),int(rect[-1].imag)),(int(rect[0].real),int(rect[0].imag)),(0,255,0),1) 
    cv2.circle(img,(int(u1.real),int(u1.imag)),5,(0,0,255),-1)
    cv2.circle(img,(int(u2.real),int(u2.imag)),5,(0,0,255),-1)
    rect = [[int(rec.real),int(rec.imag)] for rec in rect]
    lines.append([[int(u1.real),int(u1.imag)],[int(u2.real),int(u2.imag)]])
    newregs.append(rect)
  cv2.imshow("getRegionsTri",img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  # print("LINES",lines[1])
  # print("******")
  return((newregs,lines))


def mapPoint(M,pts):
  r = M.dot(np.float32([pts[0],pts[1],1]))
  r = r/r[2]
  return(np.int32(r[:-1]))

def getbaseFromGrooveLines(GrooveLines,M):
  '''
    Given Groove Lines from Grooves.GetGrooveInfo
    it will return the base for the triangle in the next step
    M : it is transform matrix extracted from getPerspective given the 
    corresponding templates between images
  '''
  Gl = GrooveLines.copy()
  bases = []
  for groove in Gl:
    line = groove[0]
    line = [mapPoint(M,pt) for pt in line]
    bases.append(line)
  return(np.array(bases))