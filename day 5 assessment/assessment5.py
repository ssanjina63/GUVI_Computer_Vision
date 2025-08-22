import numpy as np
import matplotlib.pyplot as plt 
import cv2

img1 = cv2.imread('img1.jpeg')
img2 = cv2.imread('img2.jpeg')
orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

Brute_force = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)
matches = Brute_force.match(des1,des2)

matches = sorted(matches, key=lambda x: x.distance)

result = cv2.drawMatches(img1,kp1,img2,kp2, matches[:30],None,flags=2)

plt.imshow(cv2.cvtColor(result,cv2.COLOR_BGR2RGB))
plt.show()