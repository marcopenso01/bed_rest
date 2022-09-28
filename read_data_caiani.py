import os
import numpy as np
import h5py
import cv2
import matplotlib.pyplot as plt
import pydicom # for reading dicom files
import shutil

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

#legge i dati del bed-rest di Caiani
input_folder = r'F:/Caiani_bed_rest/paz1'
output_folder = r'F:/Caiani_bed_rest/pre_process'

if not os.path.exists(output_folder):
    makefolder(output_folder)
output_folder = os.path.join(output_folder, input_folder.split('/')[-1])
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
            nx = int(data_row_img.Rows)
            ny = int(data_row_img.Columns)
            px_size = data_row_img.PixelSpacing
            slice_thick = float(data_row_img.SliceThickness)
            flag = 0

pixel_size = [float(px_size[0]),
              float(px_size[1]),
              float(slice_thick)
             ]

hdf5_file = h5py.File(os.path.join(output_folder, 'data.hdf5'), "w")

dt = h5py.special_dtype(vlen=str)
hdf5_file.create_dataset('pixel_size', (1, 3), dtype=dt)
for k in data.keys():
    hdf5_file.create_dataset(k, [len(data[k][:])] + [nx, ny], dtype=np.float32)

hdf5_file['pixel_size'][...] = pixel_size
for k in data.keys():
    for i in range(len(data[k][:])):
        hdf5_file[k][i, ...] = data[k][i]

# After loop:
hdf5_file.close()
