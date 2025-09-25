from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
from scipy.ndimage import *
from skimage.morphology import (erosion, dilation, closing, opening)
from skimage.io import imread, imshow


image=np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,255,255,255,255,255,255,0,0,0,0,0,0,0,0,0],
                [0,0,255,255,255,0,0,0,255,255,255,0,0,0,0,0,0,0,0,0],
                [0,0,255,255,255,255,0,0,255,255,255,0,0,0,0,0,0,0,0,0],
                [0,0,255,255,255,255,0,0,0,255,255,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,255,255,255,255,255,255,255,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,255,255,255,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,255,255,255,255,255,255,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,255,255,255,255,0,0,255,255,255,255,255,0,0],
                [0,0,0,0,0,0,0,255,255,255,255,255,0,0,0,0,255,255,0,0],
                [0,0,0,0,0,255,255,255,255,255,0,0,0,0,0,0,255,255,0,0],
                [0,0,0,0,255,255,255,0,0,0,255,255,255,255,255,255,255,255,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],dtype='uint8')


structuring_element = np.array([[0, 1, 0],
                                [1, 1, 1],
                                [0, 1, 0]]) 





fig, ax = plt.subplots(1,2, figsize=(15,5))
ax[0].imshow(erosion(image, structuring_element), cmap='gray')
ax[0].set_title('Eroded Image')
ax[1].imshow(dilation(image, structuring_element), cmap='gray')
ax[1].set_title('Dilated Image')

plt.show()

fig, ax = plt.subplots(1,2, figsize=(15,5))
ax[0].imshow(opening(image, structuring_element), cmap='gray')
ax[0].set_title('Opened Image')
ax[1].imshow(closing(image, structuring_element), cmap='gray')
ax[1].set_title('Closed Image')

plt.show()