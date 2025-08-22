import cv2
import matplotlib.pyplot as plt

img=cv2.imread("image.jpeg")

gaussion=cv2.GaussianBlur(img,(5,5),0)
edges = cv2.Canny(img,100,300)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


plt.imshow(cv2.cvtColor(gray,cv2.COLOR_BGR2RGB))

plt.show() 
