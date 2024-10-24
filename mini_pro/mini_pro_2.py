import cv2
import numpy as np

MAXDISP = 20
SLOP = MAXDISP + 1


def computeIntensityGradientsX(imgL, imgR, th):
    th = 5
    w = 3

    kernel = np.array([[-1, 0, 1]], dtype=np.float32)

    fil_Il = cv2.filter2D(Il, -1, kernel)
    fil_Ir = cv2.filter2D(Ir, -1, kernel)
    
    fil_Il = np.abs(fil_Il)
    fil_Ir = np.abs(fil_Ir)

    fil_Il = np.where((fil_Il <= th), 0, fil_Il)
    fil_Ir = np.where((fil_Ir <= th), 0, fil_Ir)
    for i in fil_Il:
        if i < 4:
            print(i)

    return fil_Il, fil_Ir

def computeIntensityGradientsX_2(imgL, imgR, th):
    th = 5
    w = 3

    Il = imgL.copy()
    Ir = imgL.copy()
    print(imgl.shape)
    for i in range(imgl.shape[0]-2):
        sub_Il = fil_Il[i:i+w]
        sub_Ir = fil_Il[i:i+w]
        if(th =<  sub_Il.max() - sub_Il.min()):
            Il[i] = 1000
        else
            Il[i] = 0 
        if(th =<  sub_Ir.max() - sub_Ir.min()):
            Ir[i] = 1000
        else
            Ir[i] = 0 



    return Il, Ir


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
    Il = img_0[i].copy().astype(float)
    Ir = img_2[i].copy().astype(float)
    
    pixel_disparity = np.zeros((img_0.shape[1], delta*2))
    for j in range(1, img_0.shape[1]-2):
        for k in range(j-delta,j+delta,1):
            if(k >= 0 and k <= img_0.shape[1]-1):
                mids = abs(Il[k] - Ir[j])
                pixel_disparity[j][k-j+delta] = mids
    
    cv2.imshow("img", pixel_disparity)
    #cv2.imwrite(f"pixel_disparity_{i}.png",pixel_disparity)

    Il_igr, Ir_igr = computeIntensityGradientsXi_2(Il, Ir, 5)


    for y in range(y, img_0.shape[1]-2):
        for delta_a in range(MAXDISP):
            phi_best = float("inf")
            for delta_p in range(MAXDISP):




    #cv2.waitKey(0)

cv2.imshow("img 0", img_0)
cv2.imshow("img 1", img_1)
cv2.imshow("img 2", img_2)
cv2.imshow("img 3", img_3)



cv2.waitKey(0)


"""
def fillDissimilarityTable(imgL, imgR, dis, scanline):
    {
	unsigned short int y, alpha;

	for (y = 0 ; y < g_cols ; y++)
		for (alpha = 0 ; alpha <= g_maxdisp ; alpha++)  {
			if (y+alpha < g_cols)  {
				dis[(g_maxdisp + 1)*y+alpha] = 2 * abs(imgL[g_cols*scanline+y+alpha] - imgR[g_cols*scanline+y]);
			}
		}
        for y in range(1, img_0.shape[0]-2):
}
def matchScanlines(imgL, imgR):
    fillDissimilarityTable(imgL, img R , dis, scanlines)
"""
