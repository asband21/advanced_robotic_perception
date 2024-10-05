import cv2
import numpy as np

def d_lin(xi, yi, Il, Ir):
    ir_plus = 0.5*(Ir[yi]+Ir[yi+1])
    ir_minnus = 0.5*(Ir[yi]+Ir[yi-1])
    I_min = min(ir_minnus, ir_plus, Ir[yi])
    I_min = min(ir_minnus, ir_plus, Ir[yi])
    I_max = max(ir_minnus, ir_plus, Ir[yi])
    return max(0, Il[xi]-I_max, I_min - Il[xi])


def d(xi, yi, Il, Ir):
    return min(d_lin(xi, yi, Il, Ir),d_lin(yi, xi, Ir, Il))

delta = 20 # Maximum disparity
κ_occ = 25 # Occlusion penalty
kr = 5 # Match reward
tr = 14 # Reliability threshold
a = 0.15 # Reliability buffer factor


# Open picture
img_0 = cv2.imread("data/delivery_area_1l/im0.png" , cv2.IMREAD_GRAYSCALE)
img_1 = cv2.imread("data/delivery_area_1l/im1.png" , cv2.IMREAD_GRAYSCALE)
img_2 = img_1.copy()
img_3 = cv2.absdiff(img_0, img_1)

img_0 = cv2.blur(img_0,(5,5))
img_1 = cv2.blur(img_1,(5,5))

for i in range(1, img_0.shape[0]-2):
    for j in range(1, img_0.shape[1]-2):
        Il = img_0[i].copy().astype(float)
        Ir = img_2[i].copy().astype(float)
        img_2[i][j] = int(d(i,j, Il, Ir))



cv2.imshow("img 0", img_0)
cv2.imshow("img 1", img_1)
cv2.imshow("img 2", img_2)
cv2.imshow("img 3", img_3)



cv2.waitKey(0)

