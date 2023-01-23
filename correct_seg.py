"""
Created on Fri Oct 14 11:26:46 2022
@author: mpenso
"""
import numpy as np
import h5py
import cv2
import matplotlib.pyplot as plt

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # print(x,",",y)
        X.append(y)
        Y.append(x)

drawing=False # true if mouse is pressed
mode=True

def paint_draw(event,former_x,former_y,flags,param):
    global current_former_x,current_former_y,drawing, mode
    if event==cv2.EVENT_LBUTTONDOWN:
        drawing=True
        current_former_x,current_former_y=former_x,former_y
    elif event==cv2.EVENT_MOUSEMOVE:
        if drawing==True:
            if mode==True:
                cv2.line(img,(current_former_x,current_former_y),(former_x,former_y),(255,255,255),2)
                cv2.line(image_binary,(current_former_x,current_former_y),(former_x,former_y),(255,255,255),2)
                current_former_x = former_x
                current_former_y = former_y
    elif event==cv2.EVENT_LBUTTONUP:
        drawing=False
        if mode==True:
            cv2.line(img,(current_former_x,current_former_y),(former_x,former_y),(255,255,255),2)
            cv2.line(image_binary,(current_former_x,current_former_y),(former_x,former_y),(255,255,255),2)
            current_former_x = former_x
            current_former_y = former_y
    return former_x,former_y

def imfill(img, dim):
    img = img[:,:,0]
    img = cv2.resize(img, (dim, dim))
    img[img>0]=255
    im_floodfill = img.copy()
    h, w = im_floodfill.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(im_floodfill, mask, (0,0), 255);
    return img | cv2.bitwise_not(im_floodfill)

def imfill2(img):
    im_floodfill = img.copy()
    h, w = im_floodfill.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(im_floodfill, mask, (0,0), 255);
    return img | cv2.bitwise_not(im_floodfill)

path = r'D:/BED_REST/logdir/model9/bed_rest/paz1_120/pred.hdf5'
slice_number = 3
data = h5py.File(path, "r+")
vol=[]
tit=['epicardium', 'endocardium', 'rv']
img = data['img_raw'][slice_number]
dim = img.shape[0]
img = cv2.normalize(src=img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)    

for ii in range(3):
    img = cv2.resize(img, (400, 400), interpolation = cv2.INTER_CUBIC)
    image_binary = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)

    cv2.namedWindow(tit[ii])
    cv2.setMouseCallback(tit[ii],paint_draw)
    while(1):
        cv2.imshow(tit[ii],img)
        k=cv2.waitKey(1)& 0xFF
        if k==27: #Escape KEY
            if ii==0:

                im_out1 = imfill(image_binary, dim)
                im_out1[im_out1>0]=255            
            elif ii==1:               
                im_out2 = imfill(image_binary, dim)
                im_out2[im_out2>0]=255
            elif ii==2:
                im_out3 = imfill(image_binary, dim)
                im_out3[im_out3>0]=255
            break
    cv2.destroyAllWindows()

if im_out3.sum() == imfill2(im_out3).sum():
    temp = im_out3+im_out1
    temp[temp !=0]=1
    temp = imfill2(temp)
    temp = temp-im_out1
    temp[temp != 0] = 1

vol.append(temp)
myo = im_out1 - im_out2
myo[myo!=0]=2
im_out2[im_out2!=0]=3
vol.append(myo)
vol.append(im_out2)

mask = sum(vol)
plt.figure()
plt.imshow(mask)
data['pred'][slice_number] = mask
data.close()
