"""
Created on Wed Jun 22 11:18:42 2022

@author: Marco Penso
"""

import os
import numpy as np
import h5py
import cv2
import pydicom # for reading dicom files
import matplotlib.pyplot as plt
import shutil
X = []
Y = []

def makefolder(folder):
    '''
    Helper function to make a new folder if doesn't exist
    :param folder: path to new folder
    :return: True if folder created, False if folder already exists
    '''
    if not os.path.exists(folder):
        os.makedirs(folder)
        return True
    return False


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


def crop_or_pad_slice_to_size(slice, nx, ny):
    
    if len(slice.shape) == 3:
        stack = [slice[:,:,0], slice[:,:,1], slice[:,:,2]]
        RGB = []
    else:
        stack = [slice]
    
    for i in range(len(stack)):
        
        img = stack[i]
            
        x, y = img.shape
        
        x_s = (x - nx) // 2
        y_s = (y - ny) // 2
        x_c = (nx - x) // 2
        y_c = (ny - y) // 2
    
        if x > nx and y > ny:
            slice_cropped = img[x_s:x_s + nx, y_s:y_s + ny]
        else:
            slice_cropped = np.zeros((nx, ny))
            if x <= nx and y > ny:
                slice_cropped[x_c:x_c + x, :] = img[:, y_s:y_s + ny]
            elif x > nx and y <= ny:
                slice_cropped[:, y_c:y_c + y] = img[x_s:x_s + nx, :]
            else:
                slice_cropped[x_c:x_c + x, y_c:y_c + y] = img[:, :]
        if len(stack)>1:
            RGB.append(slice_cropped)
    
    if len(stack)>1:
        return np.dstack((RGB[0], RGB[1], RGB[2]))
    else:
        return slice_cropped
    
    
def crop_or_pad_slice_to_size_specific_point(slice, nx, ny, cx, cy):
    
    if len(slice.shape) == 3:
        stack = [slice[:,:,0], slice[:,:,1], slice[:,:,2]]
        RGB = []
    else:
        stack = [slice]
        
    for i in range(len(stack)):
        img = stack[i]
        x, y = img.shape
        y1 = (cy - (ny // 2))
        y2 = (cy + (ny // 2))
        x1 = (cx - (nx // 2))
        x2 = (cx + (nx // 2))
    
        if y1 < 0:
            img = np.append(np.zeros((x, abs(y1))), img, axis=1)
            x, y = img.shape
            y1 = 0
        if x1 < 0:
            img = np.append(np.zeros((abs(x1), y)), img, axis=0)
            x, y = img.shape
            x1 = 0
        if y2 > 512:
            img = np.append(img, np.zeros((x, y2 - 512)), axis=1)
            x, y = img.shape
        if x2 > 512:
            img = np.append(img, np.zeros((x2 - 512, y)), axis=0)
    
        slice_cropped = img[x1:x1 + nx, y1:y1 + ny]
        if len(stack)>1:
            RGB.append(slice_cropped)
        
    if len(stack)>1:
        return np.dstack((RGB[0], RGB[1], RGB[2]))
    else:
        return slice_cropped


def generator_mask(img, green_pixels):
    
    flagENDO = False
    flagRV = False
    
    mask_epi = imfill(green_pixels)
    
    if len(np.argwhere(mask_epi)) == len(np.argwhere(green_pixels)):
        val = 11
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (val, val))
        result = cv2.morphologyEx(green_pixels, cv2.MORPH_CLOSE, kernel)
        while len(np.argwhere(result)) == len(np.argwhere(imfill(result))):
            val += 2
            if val > 40:
                break
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (val, val))
            result = cv2.morphologyEx(green_pixels, cv2.MORPH_CLOSE, kernel)
        mask_epi = imfill(result)
        
    red_pixels = cv2.inRange(img, (110, 0, 0), (255, 100, 100))
    
    if len(np.argwhere(red_pixels)) > 5:
        
        flagENDO = True
        
        if len(np.argwhere(red_pixels)) != len(np.argwhere(imfill(red_pixels))):
            
            mask_endo = imfill(red_pixels)  #mask LV
        else:
            val = 11
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (val, val))
            result = cv2.morphologyEx(red_pixels, cv2.MORPH_CLOSE, kernel)
            while len(np.argwhere(result)) == len(np.argwhere(imfill(result))):
                val += 2
                if val > 40:
                    break
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (val, val))
                result = cv2.morphologyEx(red_pixels, cv2.MORPH_CLOSE, kernel)
            mask_endo = imfill(result)
        
        if len(np.argwhere(green_pixels)) == len(np.argwhere(imfill(green_pixels))):
            mask_epi = imfill(mask_endo+mask_epi)

        mask_myo = mask_epi - mask_endo  #mask Myo
        
        mask_endo[mask_endo>0]=3
        
        mask_myo[mask_myo>0]=2

    yellow_pixels = cv2.inRange(img, (110, 110, 0), (255, 255, 130))
    
    if len(np.argwhere(yellow_pixels)) > 5:
        
        yellow_pixels2 = yellow_pixels.copy()
        for xx in range(len(np.argwhere(yellow_pixels))):
            coord = np.argwhere(yellow_pixels)[xx]
            if img[coord[0],coord[1],0] != img[coord[0],coord[1],1]:
                yellow_pixels2[coord[0],coord[1]] = 0
        
        flagRV = True

        temp = imfill(green_pixels + yellow_pixels2)

        mask_RV = temp - mask_epi
        
        mask_RV[mask_RV>0]=1
    
    mask_epi[mask_epi>0]=2
                
    #binary mask 0-1
    if flagENDO:
        final_mask = mask_endo + mask_myo
    else:
        final_mask = mask_epi
    if flagRV:
        final_mask += mask_RV
    
    return final_mask
 
 
def generator_mask2(img, green_pixels):
    
    flagENDO = False
    flagRV = False
    
    size = img.shape
    
    yellow_pixels = cv2.inRange(img, (110, 110, 0), (255, 255, 130))
    
    #msk = np.zeros(size[0:2], dtype=np.uint8)
    #for xx in range(len(np.argwhere(yellow_pixels))):
    #    coord = np.argwhere(yellow_pixels)[xx]
    #    if img[coord[0],coord[1],1] == img[coord[0],coord[1],2]:
    #        msk[coord[0],coord[1]] = 1
    #yellow_pixels[msk==1] = 0
    
    yellow_pixels2 = yellow_pixels.copy()
    
    if len(np.argwhere(yellow_pixels)) > 5:
        
        for xx in range(len(np.argwhere(yellow_pixels))):
            coord = np.argwhere(yellow_pixels)[xx]
            if img[coord[0],coord[1],0] != img[coord[0],coord[1],1]:
                yellow_pixels2[coord[0],coord[1]] = 0
        
        flagRV = True
    
        mask_RV = imfill(yellow_pixels2)
        
        if len(np.where(yellow_pixels2 == 255)[0]) == len(np.where(mask_RV == 255)[0]) or len(np.where(yellow_pixels2 == 255)[0])+6 >= len(np.where(mask_RV == 255)[0]):
            
            mask_RV = imfill(yellow_pixels)
            
            if len(np.where(yellow_pixels == 255)[0]) == len(np.where(mask_RV == 255)[0]) or len(np.where(yellow_pixels == 255)[0])+6 >= len(np.where(mask_RV == 255)[0]):
                return generator_mask(img, green_pixels)
            else:
                yellow_pixels2 = yellow_pixels
        
    else:
        
        mask_RV = np.zeros((size[0],size[1]), dtype=np.uint8)
    
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_RV)
    if n_labels > 2:
        val_label = np.asarray(range(n_labels))
        area = stats[:,-1]
        area, val_label = zip(*sorted(zip(area, val_label)))
        for z in range(n_labels-2):
            mask_RV[labels == val_label[z]] = 0
    
    mask_RV_MYO = imfill(green_pixels+yellow_pixels2)
    
    if len(np.argwhere(green_pixels+imfill(yellow_pixels2))) == len(np.argwhere(mask_RV_MYO)):
        val = 11
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (val, val))
        result = cv2.morphologyEx(green_pixels, cv2.MORPH_CLOSE, kernel)
        while len(np.argwhere(result)) == len(np.argwhere(imfill(result))):
            val += 2
            if val > 40:
                break
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (val, val))
            result = cv2.morphologyEx(green_pixels, cv2.MORPH_CLOSE, kernel)
        mask_RV_MYO = imfill(result)+imfill(yellow_pixels2)
    
    mask_epi = mask_RV_MYO - mask_RV
        
    red_pixels = cv2.inRange(img, (110, 0, 0), (255, 100, 100))
    
    if len(np.argwhere(red_pixels)) > 5:
        
        flagENDO = True
        
        if len(np.argwhere(red_pixels)) != len(np.argwhere(imfill(red_pixels))):
            
            mask_endo = imfill(red_pixels)  #mask LV
        else:
            val = 11
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (val, val))
            result = cv2.morphologyEx(red_pixels, cv2.MORPH_CLOSE, kernel)
            while len(np.argwhere(result)) == len(np.argwhere(imfill(result))):
                val += 2
                if val > 40:
                    break
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (val, val))
                result = cv2.morphologyEx(red_pixels, cv2.MORPH_CLOSE, kernel)
            mask_endo = imfill(result)
        
        if len(np.argwhere(green_pixels)) == len(np.argwhere(imfill(green_pixels))):
            mask_epi = imfill(mask_endo+mask_epi)
        
        mask_myo = mask_epi - mask_endo  #mask Myo
        
        mask_endo[mask_endo>0]=3
        
        mask_myo[mask_myo>0]=2
    
    if flagRV:
        
        mask_RV[mask_RV>0]=1
    
    if not flagENDO:
        
        mask_epi[mask_epi>0]=2
                
    #binary mask 0-1
    if flagENDO:
        final_mask = mask_endo + mask_myo
    else:
        final_mask = mask_epi
    if flagRV:
        final_mask += mask_RV
    
    return final_mask


def concatenate(list_file, out_fold, name_file):
    print('------- %s --------' % name_file)
    dt = h5py.special_dtype(vlen=str)
    data_file_path = os.path.join(out_fold, name_file+'.hdf5')
    hdf5_file = h5py.File(data_file_path, "w")
    c=1
    for file in list_file:
        print(file)
        data = h5py.File(os.path.join(file, 'pre_proc.hdf5'), 'r')
        d1 = data['paz'][()]
        d2 = data['phase'][()]
        d3 = data['mask'][()]
        d4 = data['img_seg'][()]
        d5 = data['img_raw'][()]
        d6 = data['img_up'][()]
        d7 = data['img_down'][()]
        d8 = data['img_left'][()]
        d9 = data['img_right'][()]
        
        print("img_raw:", d5.shape, d5.dtype)
        print("mask:", d3.shape, d3.dtype)
        if c==1:
            paz = d1
            phase = d2
            mask = d3
            img_seg = d4
            img_raw = d5
            img_up = d6
            img_down = d7
            img_left = d8
            img_right = d9
            c += 1
        else:
            paz = np.concatenate((paz, d1), axis=0)
            phase = np.concatenate((phase, d2), axis=0)
            mask = np.concatenate((mask, d3), axis=0)
            img_seg = np.concatenate((img_seg, d4), axis=0)
            img_raw = np.concatenate((img_raw, d5), axis=0)
            img_up = np.concatenate((img_up, d6), axis=0)
            img_down = np.concatenate((img_down, d7), axis=0)
            img_left = np.concatenate((img_left, d8), axis=0)
            img_right = np.concatenate((img_right, d9), axis=0)
        print("img_raw after conc:", img_raw.shape)
        print("mask after conc:", mask.shape)
        data.close()
    
    hdf5_file.create_dataset('paz', paz.shape, dtype=dt)
    hdf5_file.create_dataset('phase', phase.shape, dtype=dt)
    hdf5_file.create_dataset('mask', mask.shape, mask.dtype)
    hdf5_file.create_dataset('img_seg', img_seg.shape, img_seg.dtype)
    hdf5_file.create_dataset('img_raw', img_raw.shape, img_raw.dtype)
    hdf5_file.create_dataset('img_up', img_up.shape, img_up.dtype)
    hdf5_file.create_dataset('img_down', img_down.shape, img_down.dtype)
    hdf5_file.create_dataset('img_left', img_left.shape, img_left.dtype)
    hdf5_file.create_dataset('img_right', img_right.shape, img_right.dtype)
    
    hdf5_file['paz'][()] = paz
    hdf5_file['phase'][()] = phase
    hdf5_file['mask'][()] = mask
    hdf5_file['img_seg'][()] = img_seg
    hdf5_file['img_raw'][()] = img_raw
    hdf5_file['img_up'][()] = img_up
    hdf5_file['img_down'][()] = img_down
    hdf5_file['img_left'][()] = img_left
    hdf5_file['img_right'][()] = img_right
    
    hdf5_file.close()
 

if __name__ == '__main__':
    # legge una cartella paziente e crea le maschere 
    # Paths settings
    path = r'G:/BED-REST/file'
    nx = 160
    ny = 160
    force_overwrite = True
    crop = 200
    name = 'paz87'
    # 
    
    output_folder = os.path.join(path,name,'pre_processing')
    if not os.path.exists(output_folder) or force_overwrite:
        makefolder(output_folder)

    if not os.path.isfile(os.path.join(output_folder, 'pre_proc.hdf5')) or force_overwrite:

        print('This configuration of mode has not yet been preprocessed')
        print('Preprocessing now!')

        # ciclo su pazienti train
        MASK = []
        IMG_SEG = []  # img in uint8 con segmentazione
        IMG_RAW = []  # img in float senza segmentazione
        IMG_UP = []
        IMG_DOWN = []
        IMG_LEFT = []
        IMG_RIGHT = []
        PAZ = []
        PHS = []
        
        paz_path = os.path.join(path, name)
        
        print('---------------------------------------------------------------------------------')
        print('processing paz %s' % name)
        n_img = 0

        path_seg = os.path.join(paz_path, 'seg')
        if not os.path.exists(path_seg):
            raise Exception('path %s not found' % path_seg)
        path_seg = os.path.join(path_seg, os.listdir(path_seg)[0])

        path_raw = os.path.join(paz_path, 'raw')
        if not os.path.exists(path_raw):
            raise Exception('path %s not found' % path_raw)
        path_raw = os.path.join(path_raw, os.listdir(path_raw)[0])

        if len(os.listdir(path_seg)) != len(os.listdir(path_raw)):
            raise Exception('number of file in seg %s and row %s is not equal, for patient %s' % (
            len(os.listdir(path_seg)), len(os.listdir(path_raw)), name))

        for ff, kk in zip(os.listdir(path_seg), os.listdir(path_raw)):
            if ff != kk:
                raise Exception('file name in seg %s and row %s is not equal, for patient %s' % (ff, kk, name))


        dcmPath = os.path.join(path_raw, os.listdir(path_raw)[0])
        data_row_img = pydicom.dcmread(dcmPath)
        if data_row_img.BitsAllocated != 16:
            print('bit allocated are not 16, for patient %s' % (name))

        # select center image
        print('selec center ROI')
        X = []
        Y = []
        data_row_img = pydicom.dcmread(os.path.join(path_seg, os.listdir(path_seg)[120]))
        while True:
            img = data_row_img.pixel_array
                
            cv2.imshow("image", img.astype('uint8'))
            cv2.namedWindow('image')
            cv2.setMouseCallback("image", click_event)
            k = cv2.waitKey(0)
            plt.figure()
            plt.imshow(crop_or_pad_slice_to_size_specific_point(img, crop, crop, X[-1], Y[-1]))
            plt.show()
            # press 'q' to exit
            if k == ord('q') or k == 27:
                break
            else:
                cv2.destroyAllWindows()          
        cv2.destroyAllWindows()
            
        print('center coordinate:', X[-1], Y[-1])
        
        phase1 = []
        phase2 = []
        phase = [phase1, phase2]
        for i in range(len(os.listdir(path_seg))):
            
            if len(phase1)!=0 and len(phase2)!=0:
                n_img = int(os.listdir(path_seg)[i].split('img')[1].split('-')[0])
                if n_img % int(phase1[-1].split('img')[1].split('-')[0]) == 30 or n_img % int(phase2[-1].split('img')[1].split('-')[0]) == 30:
                    data_row_img = pydicom.dcmread(os.path.join(path_seg, os.listdir(path_seg)[i]))
                    img = data_row_img.pixel_array
                    img = crop_or_pad_slice_to_size_specific_point(img, crop, crop, X[-1], Y[-1])
                    flag=0
                    for r in range(0, img.shape[0]):
                        for c in range(0, img.shape[1]):
                            if img[r,c,0] != img[r,c,1] or img[r,c,0] != img[r,c,2]:
                                if n_img % int(phase1[-1].split('img')[1].split('-')[0]) == 30:
                                    phase1.append(os.listdir(path_seg)[i])
                                else: 
                                    phase2.append(os.listdir(path_seg)[i])
                                flag=1
                                break
                        if flag:
                            break
                else:
                    continue
            else:
                dcmPath = os.path.join(path_seg, os.listdir(path_seg)[i])
                data_row_img = pydicom.dcmread(dcmPath)
                img = data_row_img.pixel_array
                img = crop_or_pad_slice_to_size_specific_point(img, crop, crop, X[-1], Y[-1])
                flag=0
                for r in range(0, img.shape[0]):
                    for c in range(0, img.shape[1]):
                        if img[r,c,0] != img[r,c,1] or img[r,c,0] != img[r,c,2]:
                            n_img = int(os.listdir(path_seg)[i].split('img')[1].split('-')[0])
                            if len(phase1)==0:
                                phase1.append(os.listdir(path_seg)[i])
                            elif len(phase2)==0:
                                if (int(phase1[-1].split('img')[1].split('-')[0]) + 30) == n_img:
                                    phase1.append(os.listdir(path_seg)[i])
                                else:
                                    phase2.append(os.listdir(path_seg)[i])                    
                            flag=1
                            break
                    if flag:
                        break
        
        count=0
        for ph in phase:
            for i in range(len(ph)):
                dcmPath = os.path.join(path_seg, ph[i])
                #print(dcmPath)
                data_row_img = pydicom.dcmread(dcmPath)
                img = data_row_img.pixel_array
                img = crop_or_pad_slice_to_size_specific_point(img, crop, crop, X[-1], Y[-1])
                temp_img = img.copy()
                for r in range(0, img.shape[0]):
                    for c in range(0, img.shape[1]):
                        if img[r,c,0] == img[r,c,1] == img[r,c,2]:
                            temp_img[r,c,:]=0
        
                green_pixels = cv2.inRange(temp_img, (0, 110, 0), (125, 255, 125))
        
                if len(np.argwhere(green_pixels)) > 5:
                    
                    green_pixels2 = green_pixels.copy()
                    for xx in range(len(np.argwhere(green_pixels))):
                        coord = np.argwhere(green_pixels)[xx]
                        if temp_img[coord[0],coord[1],0] == temp_img[coord[0],coord[1],1]:
                            green_pixels2[coord[0],coord[1]] = 0
                    
                    final_mask = generator_mask2(temp_img, green_pixels2)
                    
                    if final_mask.max() > 3:
                        print('ERROR: max value of the mask %s is %d' % (ph[i], final_mask.max()))
                        if final_mask.max() == 4:
                            plt.figure()
                            plt.imshow(final_mask)
                            plt.title('error mask %d' % (i+1));
                            final_mask[final_mask==4]=2
                            plt.figure()
                            plt.imshow(final_mask)
                            plt.title('corrected mask %d' % (i+1));
                    
                    img = cv2.resize(img, (nx, ny), interpolation=cv2.INTER_AREA)
                    final_mask = cv2.resize(final_mask, (nx, ny), interpolation=cv2.INTER_NEAREST)
                    MASK.append(final_mask)
                    IMG_SEG.append(img)
                    
                    fig = plt.figure()
                    ax1 = fig.add_subplot(121)
                    ax1.imshow(img)
                    ax2 = fig.add_subplot(122)
                    ax2.imshow(final_mask)
                    plt.title('img %s' % (ph[i].split('-')[0]));
                    plt.show()
                
                    # save data raw
                    dcmPath = os.path.join(path_raw, ph[i])
                    data_row_img = pydicom.dcmread(dcmPath)
                    img = data_row_img.pixel_array
                    img = crop_or_pad_slice_to_size_specific_point(img, crop, crop, X[-1], Y[-1])
                    img = cv2.resize(img, (nx, ny), interpolation=cv2.INTER_AREA)
                    IMG_RAW.append(img)
                    
                    PAZ.append(name)
                    PHS.append('phase'+str(count))
                    
                # save spatial/temporal image
                for ind in range(len(os.listdir(path_raw))):
                    if os.listdir(path_raw)[ind] == ph[i]:
                        dcmPath = os.path.join(path_raw, os.listdir(path_raw)[ind - 30])
                        data_row_img = pydicom.dcmread(dcmPath)
                        img = data_row_img.pixel_array
                        img = crop_or_pad_slice_to_size_specific_point(img, crop, crop, X[-1], Y[-1])
                        img = cv2.resize(img, (nx, ny), interpolation=cv2.INTER_AREA)
                        IMG_UP.append(img)
        
                        dcmPath = os.path.join(path_raw, os.listdir(path_raw)[ind + 30])
                        data_row_img = pydicom.dcmread(dcmPath)
                        img = data_row_img.pixel_array
                        img = crop_or_pad_slice_to_size_specific_point(img, crop, crop, X[-1], Y[-1])
                        img = cv2.resize(img, (nx, ny), interpolation=cv2.INTER_AREA)
                        IMG_DOWN.append(img)
        
                        # save temporal image
                        pos_row = (ind // 30) + 1
                        pos_row = (ind // 30) + 1
                        ph1 = 30 * (pos_row - 1)
                        ph30 = 30 * (pos_row) - 1
                        vet = range(ph1, ph30 + 1)
        
                        pos_col = ind % 30
                        for _ in range(2):
                            pos_col -= 1
                            if pos_col < 0:
                                pos_col = 29
        
                        dcmPath = os.path.join(path_raw, os.listdir(path_raw)[vet[pos_col]])
                        data_row_img = pydicom.dcmread(dcmPath)
                        img = data_row_img.pixel_array
                        img = crop_or_pad_slice_to_size_specific_point(img, crop, crop, X[-1], Y[-1])
                        img = cv2.resize(img, (nx, ny), interpolation=cv2.INTER_AREA)
                        IMG_LEFT.append(img)
        
                        pos_col = ind % 30
                        for _ in range(2):
                            pos_col += 1
                            if pos_col > 29:
                                pos_col = 0
        
                        dcmPath = os.path.join(path_raw, os.listdir(path_raw)[vet[pos_col]])
                        data_row_img = pydicom.dcmread(dcmPath)
                        img = data_row_img.pixel_array
                        img = crop_or_pad_slice_to_size_specific_point(img, crop, crop, X[-1], Y[-1])
                        img = cv2.resize(img, (nx, ny), interpolation=cv2.INTER_AREA)
                        IMG_RIGHT.append(img)
                
            count+=1

        
        hdf5_file = h5py.File(os.path.join(output_folder, 'pre_proc.hdf5'), "w")

        dt = h5py.special_dtype(vlen=str)
        hdf5_file.create_dataset('paz', (len(PAZ),), dtype=dt)
        hdf5_file.create_dataset('phase', (len(PHS),), dtype=dt)
        hdf5_file.create_dataset('mask', [len(MASK)] + [nx, ny], dtype=np.uint8)
        hdf5_file.create_dataset('img_seg', [len(IMG_SEG)] + [nx, ny, 3], dtype=np.uint8)
        hdf5_file.create_dataset('img_raw', [len(IMG_RAW)] + [nx, ny], dtype=np.float32)
        hdf5_file.create_dataset('img_up', [len(IMG_UP)] + [nx, ny], dtype=np.float32)
        hdf5_file.create_dataset('img_down', [len(IMG_DOWN)] + [nx, ny], dtype=np.float32)
        hdf5_file.create_dataset('img_left', [len(IMG_LEFT)] + [nx, ny], dtype=np.float32)
        hdf5_file.create_dataset('img_right', [len(IMG_RIGHT)] + [nx, ny], dtype=np.float32)

        for i in range(len(PAZ)):
            hdf5_file['paz'][i, ...] = PAZ[i]
            hdf5_file['phase'][i, ...] = PHS[i]
            hdf5_file['mask'][i, ...] = MASK[i]
            hdf5_file['img_seg'][i, ...] = IMG_SEG[i]
            hdf5_file['img_raw'][i, ...] = IMG_RAW[i]
            hdf5_file['img_up'][i, ...] = IMG_UP[i]
            hdf5_file['img_down'][i, ...] = IMG_DOWN[i]
            hdf5_file['img_left'][i, ...] = IMG_LEFT[i]
            hdf5_file['img_right'][i, ...] = IMG_RIGHT[i]

        # After loop:
        hdf5_file.close()

    else:

        print('Already preprocessed this configuration. Loading now!')
        

'''
# split data in train-test-validation
train=[]
val=[]
fold_path = r'G:/BED-REST/file'
n_file = len(os.listdir(fold_path))
random_indices = np.arange(n_file)
np.random.shuffle(random_indices)
count=0
for i in range(n_file):
    if not os.path.isfile(os.path.join(fold_path, os.listdir(fold_path)[random_indices[i]])):
        file = os.path.join(fold_path, os.listdir(fold_path)[random_indices[i]],'pre_processing')
        if count < 40:
            val.append(file)
        else:
            train.append(file)
        count += 1

output_folder = os.path.join('G:/BED-REST', 'out')
if not os.path.exists(output_folder):
    makefolder(output_folder)
concatenate(train, output_folder, 'train')
concatenate(val, output_folder, 'val')
'''
