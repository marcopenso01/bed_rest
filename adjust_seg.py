"""
Created on Mon Jan 16 10:45:05 2023

@author: Marco Penso
"""

import scipy
import scipy.io
import os
import numpy as np
import h5py
import cv2
import matplotlib.pyplot as plt
import skimage.morphology, skimage.data
import math
import random
import pydicom

X = []
Y = []

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


# path
input_path = r'F:/BED-REST/out'
# read data
data = h5py.File(os.path.join(input_path, 'train.hdf5'), 'r')
img = data['img_raw'][0].copy()
mask = data['mask'][0].copy()
# plot data
fig = plt.figure()
ax1 = fig.add_subplot(121)
ax1.imshow(img)
ax2 = fig.add_subplot(122)
ax2.imshow(mask)
plt.show()
# structure to be modified: 1: 'RV', 2: 'Myo', 3: 'LV'
struc = 3

mask_RV = mask.copy()
mask_MYO = mask.copy()
mask_LV = mask.copy()
mask_RV[mask_RV != 1] = 0
mask_MYO[mask_MYO != 2] = 0
mask_LV[mask_LV != 3] = 0
dim = img.shape[0]

if struc == 1: 
    tit=['---Segmenting RV---']
    img = cv2.normalize(src=img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    img = cv2.resize(img, (400, 400), interpolation = cv2.INTER_CUBIC)
    image_binary = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)
    cv2.namedWindow(tit[0])
    cv2.setMouseCallback(tit[0],paint_draw)
    while(1):
        cv2.imshow(tit[0],img)
        k=cv2.waitKey(1)& 0xFF
        if k==27: #Escape KEY
            im_out = imfill(image_binary, dim)
            break              
    cv2.destroyAllWindows()
    im_out[im_out>0]=1
    
    final_mask = im_out + mask_MYO + mask_LV
    m_myo = mask_MYO.copy()
    m_lv = mask_LV.copy()
    m_myo[m_myo!=0]=1
    m_lv[m_lv!=0]=1
    mm = m_myo+m_lv
    coord = np.where((mm+im_out)>1)
    for nn in range(len(coord[0])):
        final_mask[coord[0][nn],coord[1][nn]]=1
    # plot data
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax1.imshow(img)
    ax2 = fig.add_subplot(122)
    ax2.imshow(final_mask)
    plt.show()
    
elif struc == 2:
    tit=['---epicardium---', '---endocardium---']
    for ii in range(2):
        img = cv2.normalize(src=img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
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
                elif ii==1:                                         
                    im_out2 = imfill(image_binary, dim)
                break
        cv2.destroyAllWindows()
    im_out1[im_out1>0]=1
    im_out2[im_out2>0]=1
    im_out = im_out1 - im_out2
    im_out[im_out>0]=2
    
    final_mask = im_out + mask_RV + mask_LV
    m_rv = mask_RV.copy()
    m_lv = mask_LV.copy()
    m_rv[m_rv!=0]=1
    m_lv[m_lv!=0]=1
    mm = m_rv+m_lv
    coord = np.where((mm+im_out)>2)
    for nn in range(len(coord[0])):
        final_mask[coord[0][nn],coord[1][nn]]=2
    # plot data
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax1.imshow(img)
    ax2 = fig.add_subplot(122)
    ax2.imshow(final_mask)
    plt.show()
    
elif struc == 3:
    tit=['---Segmenting LV---']
    img = cv2.normalize(src=img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    img = cv2.resize(img, (400, 400), interpolation = cv2.INTER_CUBIC)
    image_binary = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)
    cv2.namedWindow(tit[0])
    cv2.setMouseCallback(tit[0],paint_draw)
    while(1):
        cv2.imshow(tit[0],img)
        k=cv2.waitKey(1)& 0xFF
        if k==27: #Escape KEY
            im_out = imfill(image_binary, dim)
            break              
    cv2.destroyAllWindows()
    im_out[im_out>0]=3
    
    final_mask = im_out + mask_RV + mask_MYO
    m_rv = mask_RV.copy()
    m_myo = mask_MYO.copy()
    m_rv[m_rv!=0]=1
    m_myo[m_myo!=0]=1
    mm = m_rv+m_myo
    coord = np.where((mm+im_out)>3)
    for nn in range(len(coord[0])):
        final_mask[coord[0][nn],coord[1][nn]]=3
    # plot data
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax1.imshow(img)
    ax2 = fig.add_subplot(122)
    ax2.imshow(final_mask)
    plt.show()
    
else:
    raise AssertionError('Inadequate number of struc')
