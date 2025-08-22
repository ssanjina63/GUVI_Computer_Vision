import cv2
import matplotlib.pyplot as plt
img = cv2.imread("image.jpeg")
print(img.shape)
cv2.rectangle(img,(500,500),(1000,1000),(255,0,0),2)
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()