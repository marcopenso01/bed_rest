import os
import numpy as np
import h5py
import cv2
import pydicom # for reading dicom files
import matplotlib.pyplot as plt
import shutil
X = []
Y = []

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # print(x,",",y)
        X.append(y)
        Y.append(x)
        cv2.destroyAllWindows()
        

def imfill(img):
    im_floodfill = img.copy()
    h, w = im_floodfill.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(im_floodfill, mask, (0,0), 255);
    return img | cv2.bitwise_not(im_floodfill)

# questa funzione nasce per risolvere problemi nella creazione delle maschere. ALcune maschere create a livello basale possono avere dei problemi
# a causa di come è stato salvata la segmentazione manuale. QUesto codice risolve questo problema.
path = r'G:/BED-REST/paz203/pre_processing/pre_proc.hdf5'
data = h5py.File(path, "r+")
for i in range(len(data['img_seg'][:])):
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax1.imshow(data['img_seg'][i])
    ax2 = fig.add_subplot(122)
    ax2.imshow(data['mask'][i])
    plt.title('%d' % (i));
    plt.show()

# nmero dell'immagine da andare a modificare la maschera
n_img = 12
img = data['img_seg'][n_img]
# c'è il ventricolo destro nella segmentazione?
RV = 'yes'

temp_img = img.copy()
for r in range(0, img.shape[0]):
    for c in range(0, img.shape[1]):
        if img[r,c,0] == img[r,c,1] == img[r,c,2]:
            temp_img[r,c,:]=0

gray = cv2.cvtColor(temp_img, cv2.COLOR_BGR2GRAY)
for r in range(0, gray.shape[0]):
    for c in range(0, gray.shape[1]):
        if gray[r,c] != 0:
            gray[r,c] = 255

if RV == 'no':
    RV = np.zeros((gray.shape))
else:
    yellow_pixels = cv2.inRange(temp_img, (110, 110, 0), (255, 255, 130))
    nb_blobs, im_with_separated_blobs, stats, _ = cv2.connectedComponentsWithStats(yellow_pixels)
    sizes = stats[:, -1]
    sizes = sizes[1:]
    nb_blobs -= 1
    min_size = sizes.max() - 1
    yellow_pixels2 = np.zeros((yellow_pixels.shape))
    for blob in range(nb_blobs):
        if sizes[blob] >= min_size:
            yellow_pixels2[im_with_separated_blobs == blob + 1] = 255
    
    yellow_pixels2 = yellow_pixels2.astype('uint8')
    RV = imfill(yellow_pixels2)


crop = data['mask'][0].shape[0]

#red_pixels = cv2.inRange(temp_img, (110, 0, 0), (255, 100, 100))
#green_pixels = cv2.inRange(temp_img, (0, 110, 0), (125, 255, 125))
#RG = green_pixels+red_pixels
nb_blobs, im_with_separated_blobs, stats, _ = cv2.connectedComponentsWithStats(gray)
sizes = stats[:, -1]
sizes = sizes[1:]
nb_blobs -= 1
min_size = sizes.max() - 1
RG = np.zeros((gray.shape))
for blob in range(nb_blobs):
    if sizes[blob] >= min_size:
        RG[im_with_separated_blobs == blob + 1] = 255

RG = RG.astype('uint8')
'''
while True:        
    cv2.imshow("image", RG.astype('uint8'))
    cv2.namedWindow('image')
    cv2.setMouseCallback("image", click_event)
    k = cv2.waitKey(0)
    # press 'q' to exit
    if k == ord('q') or k == 27:
        break
    else:
        cv2.destroyAllWindows()          
cv2.destroyAllWindows()
'''
'''
RG_full = imfill(RG)
for k in range(2):
    temp_RG = RG.copy()
    mask = np.zeros((crop+2, crop+2), np.uint8)
    cv2.floodFill(temp_RG, mask, (X[k],Y[k]), 255);
    temp_RG = np.invert(temp_RG)
    if k==0:
        myo = RG_full-temp_RG
    else:
        lv = RG_full-temp_RG
'''

RG_full = imfill(RG)
RG_full = RG_full.astype('uint8')
'''
#if RV è attaccato al MYO
RG_full = RG_full-RV
nb_blobs, im_with_separated_blobs, stats, _ = cv2.connectedComponentsWithStats(RG_full)
sizes = stats[:, -1]
sizes = sizes[1:]
nb_blobs -= 1
min_size = sizes.max() - 1
RG_full_temp = np.zeros((RG_full.shape))
for blob in range(nb_blobs):
    if sizes[blob] >= min_size:
        RG_full_temp[im_with_separated_blobs == blob + 1] = 255
RG_full = RG_full_temp.copy()

for r in range(0, RG_full.shape[0]):
    for c in range(0, RG_full.shape[1]):
        if RG_full[r,c] == 0:
            RG[r,c] = 0
'''

temp_RG = RG.copy()
mask = np.zeros((crop+2, crop+2), np.uint8)
cv2.floodFill(temp_RG, mask, (0,0), 255);
temp_RG = np.invert(temp_RG)
temp_RG = temp_RG.astype('uint8')

nb_blobs, im_with_separated_blobs, stats, _ = cv2.connectedComponentsWithStats(temp_RG)
sizes = stats[:, -1]
sizes = sizes[1:]
nb_blobs -= 1
temp_sizes = np.sort(sizes)
min_size = temp_sizes[-2]
max_size = temp_sizes[-1]
for k in range(2):
    if k == 0:
        MYO = np.zeros((RG.shape))
        for blob in range(nb_blobs):
            if sizes[blob] == min_size:
                MYO[im_with_separated_blobs == blob + 1] = 255
    else:
        LV = np.zeros((RG.shape))
        for blob in range(nb_blobs):
            if sizes[blob] == max_size:
                LV[im_with_separated_blobs == blob + 1] = 255

plt.figure()
plt.imshow(RV)
plt.figure()
plt.imshow(LV)
plt.figure()
plt.imshow(MYO)

RV = RV.astype('uint8')
MYO = MYO.astype('uint8')
LV = LV.astype('uint8')

#LV = np.invert(LV)

RV[RV>0] = 1
MYO[MYO>0] = 2
LV[LV>0] = 3

final_mask = RV + MYO + LV
final_mask = final_mask.astype('uint8')
print(final_mask.max())
fig = plt.figure()
ax1 = fig.add_subplot(121)
ax1.imshow(temp_img)
ax2 = fig.add_subplot(122)
ax2.imshow(final_mask)
plt.show()
data['mask'][n_img] = final_mask
data.close()


data = h5py.File(path, "r")
for i in range(len(data['img_seg'][:])):
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax1.imshow(data['img_seg'][i])
    ax2 = fig.add_subplot(122)
    ax2.imshow(data['mask'][i])
    plt.title('%d' % (i));
    plt.show()
data.close()


'''
a = gray.copy()
mask = np.zeros((crop+2, crop+2), np.uint8)
cv2.floodFill(a, mask, (40,75), 255);
plt.figure()
plt.imshow(a)
b = gray.copy()
mask = np.zeros((crop+2, crop+2), np.uint8)
cv2.floodFill(b, mask, (100,60), 255);
plt.figure()
plt.imshow(b)
c = gray.copy()
mask = np.zeros((crop+2, crop+2), np.uint8)
cv2.floodFill(c, mask, (125,100), 255);
plt.figure()
plt.imshow(c)
RV = np.zeros((RG.shape))
RV = RV.astype('uint8')
LV = np.zeros((RG.shape))
LV = LV.astype('uint8')
MYO = np.zeros((RG.shape))
MYO = MYO.astype('uint8')
for rr in range(0, RV.shape[0]):
    for cc in range(0, RV.shape[1]):
        if a[rr,cc] != gray[rr,cc]:
            RV[rr,cc] = 255
for rr in range(0, LV.shape[0]):
    for cc in range(0, LV.shape[1]):
        if b[rr,cc] != gray[rr,cc]:
            LV[rr,cc] = 255
for rr in range(0, MYO.shape[0]):
    for cc in range(0, MYO.shape[1]):
        if c[rr,cc] != gray[rr,cc]:
            MYO[rr,cc] = 255
            
RV = RV.astype('uint8')
MYO = MYO.astype('uint8')
LV = LV.astype('uint8')
'''
