import cv2
import numpy as np
import matplotlib.pyplot as plt

# img = np.ones((500,500),dtype=np.uint8)*255

img = np.ones((500,500),dtype=np.uint8)
# img=cv2.imread("Day02/summer.jpg")

circle=cv2.circle(img.copy(),(250,250),250,(255,255,255),-1)
rect=cv2.rectangle(img.copy(),(30,30),(470,470),(255,255,255),-1)

# circle=cv2.circle(img.copy(),(250,250),50,(255,255,255),-1)
# rect=cv2.rectangle(img.copy(),(50,50),(100,100),(255,0,0),-1)


intersection=cv2.bitwise_xor(circle,rect,None)


plt.imshow(cv2.cvtColor(intersection,cv2.COLOR_BGR2RGB))
plt.show()