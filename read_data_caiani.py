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



if __name__ == '__main__':
    
    #funzione per processare dati bed-rest (dati senza maschera)
    #legge i dati e le croppa
    #Paths settings
    
    input_folder = r'G:/BED-REST/Caiani_bed_rest'
    output_folder = os.path.join(input_folder, 'pre_process')
    
    #set size final image
    nx = 160
    ny = 160
    force_overwrite = True
    crop = 140
    #name fold paz da processare
    name = 'paz1'
    
    
    input_folder = os.path.join(input_folder, name)
    if not os.path.exists(output_folder):
        makefolder(output_folder)
    output_folder = os.path.join(output_folder, name)
    if not os.path.exists(output_folder):
        makefolder(output_folder)
    
    n_phase = len(os.listdir((os.path.join(input_folder, os.listdir(input_folder)[0]))))
    n_slice = len(os.listdir(input_folder))
    
    # check all folders have the same number of file
    for fold in os.listdir(input_folder):
        if not len(os.listdir(os.path.join(input_folder, fold))) == n_phase:
            raise Exception('found not equal number of slices between fold %s (%s) and fold %s (%s)' % 
                            (
                                os.listdir(input_folder)[0].split('_')[-1], len(os.listdir((os.path.join(input_folder, os.listdir(input_folder)[0])))),
                                fold.split('_')[-1], len(os.listdir(os.path.join(input_folder, fold)))
                            )
                           )
    
    data = {}
    for i in range(n_phase):
        data['phase'+str(i)]=[]
    
    flag = 1
    
    for fold in os.listdir(input_folder):
        fold_path = os.path.join(input_folder, fold)
        for file, i in zip(os.listdir(fold_path), range(len(os.listdir(fold_path)))):            
            dcmPath = os.path.join(fold_path, file)
            data_row_img = pydicom.dcmread(dcmPath)
            data['phase'+str(i)].append(data_row_img.pixel_array)
            if flag:
                px_size = data_row_img.PixelSpacing
                slice_thick = float(data_row_img.SliceThickness)
                flag = 0
    
    pixel_size = [float(px_size[0]),
                  float(px_size[1]),
                  float(slice_thick)
                 ]
    
    #crop images
    print('selec center ROI')
    X = []
    Y = []
    while True:
        img = data['phase10'][10]
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
    
    hdf5_file = h5py.File(os.path.join(output_folder, 'data.hdf5'), "w")
    
    dt = h5py.special_dtype(vlen=str)
    hdf5_file.create_dataset('pixel_size', (1, 3), dtype=dt)
    for k in data.keys():
        hdf5_file.create_dataset(k, [len(data[k][:])] + [nx, ny], dtype=np.float32)
        
    hdf5_file['pixel_size'][...] = pixel_size
    for k in data.keys():
        for i in range(len(data[k][:])):
            img = crop_or_pad_slice_to_size_specific_point(data[k][i], crop, crop, X[-1], Y[-1])
            img = cv2.resize(img, (nx, ny), interpolation=cv2.INTER_AREA)
            hdf5_file[k][i, ...] = img
    
    # After loop:
    hdf5_file.close()
