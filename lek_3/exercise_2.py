import cv2
import numpy as np

cap = cv2.VideoCapture('slow_traffic_small_2.mp4')
ret, frame = cap.read()
ll = frame.shape
flow_map = np.zeros((ll[0]-3, ll[1]-1,3), np.uint8)
kernel = np.ones((5,5),np.float32)/25
iii = 1
while cap.isOpened():
    print(iii)
    iii = iii + 1
    ret_gam = ret
    gam = frame
    ll = frame.shape
    ret, frame = cap.read()

    frame = cv2.filter2D(frame,-1,kernel)
    
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    img_dif =  frame - gam
    sob_x = cv2.Sobel(frame, -1 ,1 ,0, ksize=1)
    sob_y = cv2.Sobel(frame, -1, 0, 1, ksize=1)

    for i in range(ll[0]-3):
        for j in range(ll[1]-3):
            brik_x = sob_x[i:i+3, j:j+3].flatten()
            brik_y = sob_y[i:i+3, j:j+3].flatten()
            xy = np.array([brik_x,brik_y]).T
            brik_dif =img_dif[i:i+3, j:j+3].flatten()

            x, residuals, rank, s = np.linalg.lstsq(xy, brik_dif, rcond=None)
            
            flow_map[i, j, 0] = x[0]
            flow_map[i, j, 1] = x[1]
    flow_map = flow_map/500
    cv2.imshow('frame', frame)
    cv2.imshow('flow map', flow_map)
    cv2.imshow('frame sob x', sob_x)
    cv2.imshow('frame sob y', sob_y)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
