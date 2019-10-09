import numpy as np
import pywt
from matplotlib import pyplot as plt
from pywt._doc_utils import wavedec2_keys, draw_2d_wp_basis
import sys
import cv2


def visualise(l):
  c = []
  c.append(l[0])
  for detail_level in range(len(l)-1):
    c.append([np.copy(d) for d in l[detail_level + 1]])
  fig, axes = plt.subplots(1, 1, figsize=[14, 8])
  c[0] = (c[0] - np.amin(c[0]))/(np.amax(c[0]) - np.amin(c[0]))
  for detail_level in range(level):
    c[detail_level + 1] = [(d - np.amin(d))/(np.amax(d) - np.amin(d)) for d in c[detail_level + 1]]
  arr, slices = pywt.coeffs_to_array(c)
  axes.imshow(arr, cmap=plt.cm.gray)
  plt.tight_layout()
  plt.show()


# x = pywt.data.camera().astype(np.float32)
x = cv2.imread(sys.argv[1],cv2.IMREAD_GRAYSCALE)
x = x/255.0
shape = x.shape
level = 3

c = pywt.wavedec2(x, 'haar', mode='symmetric', level=level)
visualise(c)
c[0] = 0*(c[0])

losedetails = 0 #lose details till this level
assert(len(c)>losedetails)
for i in range(losedetails):
  c[len(c)- 1 - i] = [ 0*d for d in c[len(c)- 1 - i]]
x_ = pywt.waverec2(c, 'haar')
# x_ = (x_ - np.amin(c[0]))/(np.amax(x_) - np.amin(x_))

cv2.imshow("Original",x)
cv2.imshow("processed",x_)
cv2.waitKey(0)
cv2.destroyAllWindows()



# max_lev = 3       # how many levels of decomposition to draw
# label_levels = 3  # how many levels to explicitly label on the plots


# fig, axes = plt.subplots(2, 4, figsize=[14, 8])
# for level in range(0, max_lev + 1):
#     if level == 0:
#         # show the original image before decomposition
#         axes[0, 0].set_axis_off()
#         axes[1, 0].imshow(x, cmap=plt.cm.gray)
#         axes[1, 0].set_title('Image')
#         axes[1, 0].set_axis_off()
#         continue

#     # plot subband boundaries of a standard DWT basis
#     # draw_2d_wp_basis(shape, wavedec2_keys(level), ax=axes[0, level],
#     #                  label_levels=label_levels)
#     axes[0, level].set_title('{} level\ndecomposition'.format(level))

#     # compute the 2D DWT
#     c = pywt.wavedec2(x, 'db2', mode='periodization', level=level)
#     # normalize each coefficient array independently for better visibility
#     c[0] /= np.abs(c[0]).max()
#     for detail_level in range(level):
#         c[detail_level + 1] = [d/np.abs(d).max() for d in c[detail_level + 1]]
#     # show the normalized coefficients
#     arr, slices = pywt.coeffs_to_array(c)
#     axes[1, level].imshow(arr, cmap=plt.cm.gray)
#     axes[1, level].set_title('Coefficients\n({} level)'.format(level))
#     axes[1, level].set_axis_off()

# plt.tight_layout()
# plt.show()