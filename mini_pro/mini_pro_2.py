import cv2
import numpy as np

def fillDissimilarityTable():
    pass

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

    zo = np.zeros((SLOP, 1))
    Il = imgL.copy()
    Ir = imgL.copy()

    #print(Il.shape)
    Il = np.append(Il, zo)
    Ir = np.append(Ir, zo)

    #print(Il.shape)
    for i in range(Il.shape[0]-2):
        sub_Il = Il[i:i+w]
        sub_Ir = Ir[i:i+w]
        if(th <=  sub_Il.max() - sub_Il.min()):
            Il[i] = 0
        else:
            Il[i] = 1000 
        if(th <=  sub_Ir.max() - sub_Ir.min()):
            Ir[i] = 0
        else:
            Ir[i] = 1000 

    return Il, Ir

DISCONTINUITY = 255
NO_DISCONTINUITY = 0
FIRST_MATCH = 65535
DEFAULT_COST = 600
MAXDISP = 20
SLOP = MAXDISP + 1

reward = 5 * 2
occ_pen = 25 * 2 # This is the occlusion penalty.  

delta = 20 # Maximum disparity
κ_occ = 25 # Occlusion penalty
kr = 5 # Match reward
tr = 14 # Reliability threshold
a = 0.15 # Reliability buffer factor

# Open picture
#img_0 = cv2.imread("data/delivery_area_1l/im0.png" , cv2.IMREAD_GRAYSCALE)
#img_1 = cv2.imread("data/delivery_area_1l/im1.png" , cv2.IMREAD_GRAYSCALE)
img_0 = cv2.imread("b1.pgm" , cv2.IMREAD_GRAYSCALE)
img_1 = cv2.imread("b2.pgm" , cv2.IMREAD_GRAYSCALE)

img_2 = img_1.copy()
img_3 = cv2.absdiff(img_0, img_1)

img_0 = cv2.blur(img_0,(5,5))
img_1 = cv2.blur(img_1,(5,5))

#depth_discontinuities = img_0.copy() * 0
#disparity_map = img_0.copy()

depth_discontinuities = np.zeros((img_0.shape[0] + SLOP, img_0.shape[1] + SLOP))
disparity_map = np.zeros((img_0.shape[0] + SLOP, img_0.shape[1] + SLOP))

phi = np.zeros((img_0.shape[1] + SLOP, MAXDISP + 1))
pie_y = np.zeros((img_0.shape[1] + SLOP, MAXDISP + 1))
pie_d = np.zeros((img_0.shape[1] + SLOP, MAXDISP + 1))


print(f"{(img_0.shape[1] + SLOP, MAXDISP + 1)} = (img_0.shape[1] + SLOP, MAXDISP + 1)")

for i in range(0, img_0.shape[0]-1):
    Il = img_0[i].copy().astype(float)
    Ir = img_2[i].copy().astype(float)
    
    pixel_disparity = np.zeros((img_0.shape[1], delta*2))
    for j in range(0, img_0.shape[1]-1):
        for k in range(0, MAXDISP):
            if(j + k < img_0.shape[1]):
                pixel_disparity[j][k] = 2 * abs(Il[k+k] - Ir[j])
    """
         for (y = 0 ; y < COLS ; y++)
          for (alpha = 0 ; alpha <= MAXDISP ; alpha++)  {
              if (y+alpha < COLS)  {
                  dis[y][alpha] = 2 * abs(imgL[scanline][y+alpha] - imgR[scanline][y]);
              }
          }
    """
    
    cv2.imshow("img", pixel_disparity)
    #cv2.imwrite(f"pixel_disparity_{i}.png",pixel_disparity)

    Il_igr, Ir_igr = computeIntensityGradientsX_2(Il, Ir, 5)

    """
    for j in Il_igr:
        print(j)
    print("")
    print("----------------------------------------")
    print("")
    for j in Ir_igr:
        print(j)
    """

    for j in range(phi.shape[1]-1):
        phi[0][j] = DEFAULT_COST + pixel_disparity[0][j]
        pie_y[0][j] = FIRST_MATCH
        pie_d[0][j] = FIRST_MATCH
    
    #for delta_p in range(MAXDISP + 1):
    #    phi[0][delta_p] = DEFAULT_COST + dis[0][delta_p]
    pie_y_best = 1000
    pie_d_best = 1000

    for y in range(0, img_0.shape[1]-1):
        for delta_a in range(MAXDISP):
            phi_best = float("inf")
            for delta_p in range(MAXDISP):
                y_p = y - max(1, delta_p - delta_a + 1)
                if y_p >= 0:
                    #if delta_a == delta_p or                      skal implamer når for stødet er nok nogrt man synmaik programers ikke fyldige fælder
                    if delta_a == delta_p or (delta_a > delta_p and 0  == Il_igr[y + delta_a - 1]) or (delta_a < delta_p and 0 == Ir_igr[y_p+1]): 
                        phi_new = phi[y_p][delta_p] + occ_pen * (delta_a != delta_p)
                        #print(f"{phi[y_p][delta_p] + occ_pen * (delta_a != delta_p)} = {phi[y_p][delta_p]} + {occ_pen} * {(delta_a != delta_p)}")
                        if (phi_new < phi_best):
                            phi_best = phi_new
                            pie_y_best = y_p
                            pie_d_best = delta_p

        phi[y][delta_a] = phi_best + pixel_disparity[y][delta_a] - reward
        pie_y[y][delta_a] = pie_y_best
        pie_d[y][delta_a] = pie_d_best




    print(f"scanline:{i}")
    pie_d_best = np.argmin(phi[i])
    phi_best = phi[i][pie_d_best];
    """
    phi_best = INF;
    for (deltaa = 0 ; deltaa <= MAXDISP ; deltaa++)  {
      if (phi[y][deltaa] <= phi_best)  {
        phi_best = phi[y][deltaa];
        pie_y_best = y;
        pie_d_best = deltaa;
      }
    }
    """


    #cv2.waitKey(0)

    for j in range(img_0.shape[1]-1 , 1):
        disparity_map[i][j] = pie_d_best
        depth_discontinuities[i][j] = NO_DISCONTINUITY

    y2 = pie_y[i][pie_d_best]
    

    y1 =        int(pie_y_best)
    deltaa1 =   int(pie_d_best)
    x1 =        int(y1 + deltaa1)
    y2 =        int(pie_y[y1][deltaa1])
    deltaa2 =   int(pie_d[y1][deltaa1])
    x2 = y2 +   int(deltaa2)


    jjj = 00
    print("...............................")
    while y2 != FIRST_MATCH:
        print(y2)

        if deltaa1 == deltaa2:
            print(disparity_map.shape)
            print(f"disparity_map[{i}][{x2}]")
            disparity_map[i][x2] = deltaa2
            depth_discontinuities[i][x2] = NO_DISCONTINUITY;
        elif deltaa2 > deltaa1:
            disparity_map[i][x2] = deltaa2
            depth_discontinuities[i][x2] = DISCONTINUITY
        else:
            disparity_map[i][x1 - 1] = deltaa2
            depth_discontinuities[i][x1 - 1] = DISCONTINUITY
            #for (x = x1 - 2 ; x >= x2 ; x--):
            for x in range(x1-2, x2-1, -1): 
                disparity_map[i][x] = deltaa2
                depth_discontinuities[i][x] = NO_DISCONTINUITY
        
        y1 =        int(y2)
        deltaa1 =   int(deltaa2)
        x1 =        int(y1 + deltaa1)
        y2 =        int(pie_y[int(y1)][int(deltaa1)])
        deltaa2 =   int(pie_d[int(y1)][int(deltaa1)])
        x2 =        int(y2 + deltaa2)
        
        #print(f" {y2} = pie_y[{y1}][{deltaa1}]")

    """
    {
      int x, x1, x2, y1, y2, deltaa1, deltaa2;
      
      y1 = pie_y_best;         deltaa1 = pie_d_best;         x1 = y1 + deltaa1;
      y2 = pie_y[y1][deltaa1]; deltaa2 = pie_d[y1][deltaa1]; x2 = y2 + deltaa2;
      
      for (x = COLS - 1 ; x >= x1 ; x--)  {
        disparity_map[scanline][x] = deltaa1;
        depth_discontinuities[scanline][x] = NO_DISCONTINUITY;
      }
      
      while (y2 != FIRST_MATCH)  {
        if (deltaa1 == deltaa2)  {
          disparity_map[scanline][x2] = deltaa2;
          depth_discontinuities[scanline][x2] = NO_DISCONTINUITY;
        }
        else if (deltaa2 > deltaa1)  {
          disparity_map[scanline][x2] = deltaa2;
          depth_discontinuities[scanline][x2] = DISCONTINUITY;
        }
        else {
          disparity_map[scanline][x1 - 1] = deltaa2;
          depth_discontinuities[scanline][x1 - 1] = DISCONTINUITY;
          for (x = x1 - 2 ; x >= x2 ; x--)  {
            disparity_map[scanline][x] = deltaa2;
            depth_discontinuities[scanline][x] = NO_DISCONTINUITY;
          }
        }
        y1 = y2;                 deltaa1 = deltaa2;            x1 = y1 + deltaa1;
        y2 = pie_y[y1][deltaa1]; deltaa2 = pie_d[y1][deltaa1]; x2 = y2 + deltaa2;
      }
      
      for (x = y1 + deltaa1 - 1 ; x >= 0 ; x--)  {
        disparity_map[scanline][x] = deltaa1;
        depth_discontinuities[scanline][x] = NO_DISCONTINUITY;
      }
    }


    """

cv2.imshow("phi", phi)
cv2.imshow("pie_y", pie_y)
cv2.imshow("pie_d", pie_d)

cv2.imshow("disparity_map", disparity_map)
cv2.imshow("depth_discontinuities", depth_discontinuities)

cv2.imwrite("disparity_map.png", disparity_map)
cv2.imwrite("depth_discontinuities.png", depth_discontinuities)


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
