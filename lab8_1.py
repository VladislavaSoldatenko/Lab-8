import cv2
import numpy as np

img = cv2.imread('variant-5.jpg')

noise = np.random.normal(0, 25, img.shape).astype(np.uint8)
noisy_img = cv2.add(img, noise)

cv2.imwrite('noisy_variant-5.jpg', noisy_img)
cv2.imshow('Original', img)
cv2.imshow('Noisy Image', noisy_img)
cv2.waitKey(0)
cv2.destroyAllWindows()