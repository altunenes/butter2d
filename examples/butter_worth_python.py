#This for the compare outputs
from skimage.filters import butterworth
import matplotlib.pyplot as plt
import cv2
cutoff_skimage = 0.1 
img = cv2.imread("images/astronaut_gray.png",0)
img_filt_skimage = butterworth(img, cutoff_frequency_ratio=cutoff_skimage, high_pass=True, order=2)
plt.imsave("output/astronaut_gray_filtered_python.png", img_filt_skimage, cmap='gray')