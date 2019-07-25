import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster import vq

lower_blue = np.array([110,100,100])
upper_blue = np.array([130,255,255])

lower_red = np.array([170,50,50])
upper_red = np.array([180,255,255])

lower_green = np.array([65,60,60])
upper_green = np.array([80,255,255])

lower_yellow= np.array([20,100,100])
upper_yellow= np.array([30,255,255])

img = cv2.imread('colorcir.png',cv2.IMREAD_COLOR)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
mask_B = cv2.inRange(hsv, lower_blue, upper_blue)
mask_R = cv2.inRange(hsv, lower_red, upper_red)
mask_G = cv2.inRange(hsv, lower_green, upper_green)
mask_Y = cv2.inRange(hsv, lower_yellow, upper_yellow)


#img0 = cvtColor(mask, imHSV, CV_BGR2GRAY)



img1_B = cv2.medianBlur(mask_B,5)
cimg_B = cv2.cvtColor(img1_B,cv2.COLOR_GRAY2BGR)


circles_1_B = cv2.HoughCircles(mask_B,cv2.HOUGH_GRADIENT,1,20,
                            param1=10,param2=10,minRadius=0,maxRadius=100)
#print(circles_1_B[0].shape)
#print(circles_1_B[0,:,:-1].shape)
#circles=vq.kmeans(circles_1[0,:,:-1],6,iter=20)
circles_B=circles_1_B[:,:-1]
x_B = np.array(circles_B)
print(x_B[0,:,0]-635)
print(-(x_B[0,:,1]-360))
print(x_B[0,:,2])
circles_B=circles_B[0]
circles_B = np.uint16(np.around(circles_B))
for i in circles_B:
    # draw the outer circle
    #cv2.circle(cimg,(i[0],i[1]),10,(0,255,0),2)
    # draw the center of the circle
    cv2.circle(cimg_B,(i[0],i[1]),2,(0,0,255),3)

plt.imshow(cimg_B)
plt.show()

##########

img1_R = cv2.medianBlur(mask_R,5)
cimg_R = cv2.cvtColor(img1_R,cv2.COLOR_GRAY2BGR)


circles_1_R = cv2.HoughCircles(mask_R,cv2.HOUGH_GRADIENT,1,20,
                            param1=10,param2=10,minRadius=0,maxRadius=100)
#print(circles_1_R[0].shape)
#print(circles_1_R[0,:,:-1].shape)
#circles=vq.kmeans(circles_1[0,:,:-1],6,iter=20)
circles_R=circles_1_R[:,:-1]
x_R = np.array(circles_R)
print(x_R[0,:,0]-635)
print(-(x_R[0,:,1]-360))
print(x_R[0,:,2])
print("passed")
circles_R=circles_R[0]
circles_R = np.uint16(np.around(circles_R))
for i in circles_R:
    # draw the outer circle
    #cv2.circle(cimg,(i[0],i[1]),10,(0,255,0),2)
    # draw the center of the circle
    cv2.circle(cimg_R,(i[0],i[1]),2,(0,0,255),3)

plt.imshow(cimg_R)
plt.show()


img1_G = cv2.medianBlur(mask_G,5)
cimg_G = cv2.cvtColor(img1_G,cv2.COLOR_GRAY2BGR)


circles_1_G = cv2.HoughCircles(mask_G,cv2.HOUGH_GRADIENT,1,20,
                            param1=10,param2=10,minRadius=0,maxRadius=100)
#print(circles_1_G[0].shape)
#print(circles_1_G[0,:,:-1].shape)
#circles=vq.kmeans(circles_1[0,:,:-1],6,iter=20)
circles_G=circles_1_G[:,:-1]
x_G = np.array(circles_G)
print(x_G[0,:,0]-635)
print(-(x_G[0,:,1]-360))
print(x_G[0,:,2])
circles_G=circles_G[0]
circles_G = np.uint16(np.around(circles_G))
for i in circles_G:
    # draw the outer circle
    #cv2.circle(cimg,(i[0],i[1]),10,(0,255,0),2)
    # draw the center of the circle
    cv2.circle(cimg_G,(i[0],i[1]),2,(0,0,255),3)

plt.imshow(cimg_G)
plt.show()
#########
img1_Y = cv2.medianBlur(mask_Y,5)
cimg_Y = cv2.cvtColor(img1_Y,cv2.COLOR_GRAY2BGR)


circles_1_Y = cv2.HoughCircles(mask_Y,cv2.HOUGH_GRADIENT,1,20,
                            param1=10,param2=10,minRadius=0,maxRadius=100)
#print(circles_1_Y[0].shape)
#print(circles_1_Y[0,:,:-1].shape)
#circles=vq.kmeans(circles_1[0,:,:-1],6,iter=20)
circles_Y=circles_1_Y[:,:-1]
x_R = np.array(circles_Y)
print(x_R[0,:,0]-635)
print(-(x_R[0,:,1]-360))
print(x_R[0,:,2])
print("passed")
circles_Y=circles_Y[0]
circles_Y = np.uint16(np.around(circles_Y))
for i in circles_R:
    # draw the outer circle
    #cv2.circle(cimg,(i[0],i[1]),10,(0,255,0),2)
    # draw the center of the circle
    cv2.circle(cimg_Y,(i[0],i[1]),2,(0,0,255),3)

plt.imshow(cimg_Y)
plt.show()
