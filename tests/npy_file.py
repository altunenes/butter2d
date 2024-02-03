import numpy as np
from skimage.filters import butterworth
from skimage.io import imsave, imread
from skimage.color import rgb2gray
from skimage import data
img = rgb2gray(data.astronaut())
cutoff = 0.1  #
high_pass = True
order = 2
filtered_img = butterworth(img, cutoff_frequency_ratio=cutoff, high_pass=high_pass, order=order)
filtered_img_ubyte = (filtered_img * 255).astype(np.uint8)
np.save('tests/filtered_img_skimage.npy', filtered_img_ubyte)
input_img_ubyte = (img * 255).astype(np.uint8)
imsave('tests/astronaut_gray.png', input_img_ubyte)