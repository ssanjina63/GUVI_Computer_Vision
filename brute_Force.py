import cv2
import matplotlib.pyplot as plt

img1=cv2.imread("img1.jpeg")
img2=cv2.imread("img2.jpeg")

#ORB Keypoint Extractor
orb=cv2.ORB_create()
kpt1,des1= orb.detectAndCompute(img1,None)
kpt2,des2= orb.detectAndCompute(img2,None)

Brute_force=cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)
matches=Brute_force.match(des1,des2)


matches=sorted(matches,key=lambda x: x.distance)

result=cv2.drawMatches(img1,kpt1,img2,kpt2,matches[:30],None,flags=2)

plt.imshow(cv2.cvtColor(result,cv2.COLOR_BGR2GRAY))
plt.show()                                                                                                                                     

