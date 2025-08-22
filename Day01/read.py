import cv2
import matplotlib.pyplot as plt

img = cv2.imread("images.jfif")
print(img)





m=cv2.rectangle(img,(50,50),(100,100),(255,0,0),2)
# cv2.imwrite("output.jfif",img)

plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()
