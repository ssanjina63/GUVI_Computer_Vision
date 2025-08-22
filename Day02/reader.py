import cv2
import matplotlib.pyplot as plt
import numpy as np

# img=cv2.imread("Day02/images.jfif")
# print(img)
# print(img.shape)

# h,w = img.shape[:2]
# m=cv2.getRotationMatrix2D((w/12,h/12),45,1)
# rotate=cv2.warpAffine(img,m,(w,h))
# plt.imshow(cv2.cvtColor(rotate,cv2.COLOR_BGR2RGB))

# circle=cv2.circle(img,(100,100),50,(0,0,225),2)
# plt.imshow(cv2.cvtColor(circle,cv2.COLOR_BGR2RGB))


# tm=np.float32([[1,0,-50],[0,1,20]])
# translated=cv2.warpAffine(img,tm,(w,h))
# plt.imshow(cv2.cvtColor(translated,cv2.COLOR_BGR2RGB))
# plt.axis("off")
# plt.show()
# # cv2.imwrite("output.jfif",img)




# img=cv2.imread("Day02/Noisy-image.png")
# blur=cv2.blur(img,(5,5))

# gaussian=cv2.GaussianBlur(img,(5,5),0)

# median=cv2.medianBlur(img,3)


img=cv2.imread("Day02/summer.jpg")
gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
edges=cv2.Canny(img,50,100)
# threshold,thresh=cv2.threshold(gray,0,255,cv2.THRESH_BINARY)
threshold,thresh=cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
hist = cv2.calcHist(gray,[0],None,[256],[0,256])


# print(img)
# plt.imshow(cv2.cvtColor(thresh,cv2.COLOR_BGR2RGB))
plt.plot(hist)
# plt.axis("off")
plt.show()