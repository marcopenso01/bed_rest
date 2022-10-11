import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1"
# for GPU process:
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
# for CPU process:
# os.environ["CUDA_VISIBLE_DEVICES"] =

import numpy as np
import h5py
import time
import tensorflow as tf
from tensorflow.keras import backend as K
import logging
import model_structure
import losses
import metrics
import pandas as pd
import cv2
from skimage import color
from matplotlib.backends.backend_pdf import PdfPages
from tensorflow.python.client import device_lib
import matplotlib.pyplot as plt
from skimage import io, color
from skimage import measure

logging.basicConfig(
    level=logging.INFO  # allow DEBUG level messages to pass through the logger
)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

assert 'GPU' in str(device_lib.list_local_devices())
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print('is_gpu_available: %s' % tf.test.is_gpu_available())  # True/False
# Or only check for gpu's with cuda support
print('gpu with cuda support: %s' % tf.test.is_gpu_available(cuda_only=True))
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))


# tf.config.list_physical_devices('GPU') #The above function is deprecated in tensorflow > 2.1

def standardize_image(image):
    '''
    make image zero mean and unit standard deviation
    '''

    img_o = np.float32(image.copy())
    m = np.mean(img_o)
    s = np.std(img_o)
    return np.divide((img_o - m), s)


def normalize_image(image):
    '''
    make image normalize between 0 and 1
    '''
    img_o = np.float32(image.copy())
    img_o = (img_o - img_o.min()) / (img_o.max() - img_o.min())
    return img_o


def myprint(s):
    with open('modelsummary.txt', 'w+') as f:
        print(s, file=f)


def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x


def print_txt(output_dir, stringa):
    out_file = os.path.join(output_dir, 'summary_report.txt')
    with open(out_file, "a") as text_file:
        text_file.writelines(stringa)


def compute_metrics_on_directories_raw(input_fold):
    '''
    - Predicted volume
    :return: Pandas dataframe with all measures in a row for each prediction and each structure
    '''
    data = h5py.File(input_fold, 'r')

    cardiac_phase = []
    file_names = []
    structure_names = []
    slice_number = []

    # measures per structure:
    vol_list = []

    structures_dict = {1: 'RV', 2: 'Myo', 3: 'LV'}

    paz = data['paz'][0]
    for ph in np.unique(data['phase'][:]):
        ind = np.where(data['phase'][:] == ph)

        pred_arr = []  # predizione del modello

        for i in range(len(ind[0])):
            pred_arr.append(data['pred'][ind[0][i]])

        pred_arr = np.transpose(np.asarray(pred_arr, dtype=np.uint8), (1, 2, 0))

        for struc in [3, 1, 2]:
            for nslice in range(pred_arr.shape[2]):
                slice_pred = pred_arr[:,:,nslice]
                pred_binary = (slice_pred == struc) * 1

                # vol[ml] = n_pixel * (x_dim*y_dim) * z_dim / 1000
                # 1 mm^3 = 0.001 ml
                volpred = pred_binary.sum() * (float(data['pixel_size'][0][0]) * float(data['pixel_size'][0][1])) * \
                          float(data['pixel_size'][0][2]) / 1000.

                vol_list.append(volpred)  # volume predetto CNN
                slice_number.append(nslice)
                cardiac_phase.append(ph)
                file_names.append(paz)
                structure_names.append(structures_dict[struc])

    df = pd.DataFrame({'vol': vol_list, 'phase': cardiac_phase, 'struc': structure_names,
                       'filename': file_names, 'slicenumber': slice_number})
    data.close()
    return df


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
PATH
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
log_root = 'D:\BED_REST\logdir'
experiment_name = 'model7'
test_data_path = 'G:\BED-REST\Caiani_bed_rest\pre_process'

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
TESTING AND EVALUATING THE MODEL
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
print('-' * 50)
print('Testing...')
print('-' * 50)
out_path = os.path.join(log_root, experiment_name, 'bed_rest')
if not tf.io.gfile.exists(out_path):
    tf.io.gfile.makedirs(out_path)

logging.info('\n------- loading model -----------')
model = tf.keras.models.load_model(os.path.join(log_root, experiment_name, 'model_weights.h5'),
                                   custom_objects={'loss_function': losses.combo_loss(),
                                                   'average_dice_coef': losses.average_dice_coef})

total_time = 0
total_volumes = 0

for paz in os.listdir(test_data_path):

    start_time = time.time()

    logging.info(' --------------------------------------------')
    logging.info('------- Analysing paz: %s' % paz)
    logging.info(' --------------------------------------------')

    fold_paz = os.path.join(test_data_path, paz)

    data = h5py.File(os.path.join(fold_paz, 'data.hdf5'), 'r')

    test_img = []
    test_left = []
    test_right = []
    test_paz = []
    test_pred = []
    test_phase = []
    test_slice = []
    px_dim = data['pixel_size'][0]

    max_phase = 0
    for k in data.keys():
        if k == 'pixel_size':
            continue
        n_phase = int(k.split('phase')[-1])
        if n_phase > max_phase:
            max_phase = n_phase

    for k in data.keys():
        if k == 'pixel_size':
            continue
        n_phase = int(k.split('phase')[-1])
        pos_right = n_phase + 1
        pos_left = n_phase - 1
        if pos_right > max_phase:
            pos_right = 0
        if pos_left < 0:
            pos_left = max_phase
        for ii in range(len(data[k])):
            test_img.append(data[k][ii].astype('float32'))
            test_left.append(data[str('phase' + str(pos_left))][ii].astype('float32'))
            test_right.append(data[str('phase' + str(pos_right))][ii].astype('float32'))
            test_phase.append(n_phase)
            test_paz.append(paz)
            test_slice.append(ii)

    data.close()

    for ii in range(len(test_img)):
        img = test_img[ii]
        img_left = test_left[ii]
        img_right = test_right[ii]

        img = np.float32(normalize_image(img))
        img_left = np.float32(normalize_image(img_left))
        img_right = np.float32(normalize_image(img_right))

        x = np.reshape(img, (1, img.shape[0], img.shape[1], 1))
        x1 = np.reshape(img_left, (1, img.shape[0], img.shape[1], 1))
        x2 = np.reshape(img_right, (1, img.shape[0], img.shape[1], 1))

        mask_out = model.predict((x, x1, x2))
        mask_out = np.squeeze(mask_out)
        mask_out = np.argmax(mask_out, axis=-1)
        test_pred.append(mask_out)

    elapsed_time = time.time() - start_time
    total_time += elapsed_time
    total_volumes += 1
    logging.info('Evaluation of volume took %f secs.' % elapsed_time)

    if not tf.io.gfile.exists(os.path.join(out_path, paz)):
        tf.io.gfile.makedirs(os.path.join(out_path, paz))

    n_file = len(test_img)
    out_file = h5py.File(os.path.join(out_path, paz, 'pred.hdf5'), "w")

    dt = h5py.special_dtype(vlen=str)
    out_file.create_dataset('img_raw', [n_file] + [160, 160], dtype=np.float32)
    out_file.create_dataset('pred', [n_file] + [160, 160], dtype=np.uint8)
    out_file.create_dataset('phase', (n_file,), dtype=dt)
    out_file.create_dataset('paz', (n_file,), dtype=dt)
    out_file.create_dataset('pixel_size', (1,3), dtype=dt)
    out_file.create_dataset('nslice', (n_file,), dtype=dt)

    for i in range(n_file):
        out_file['img_raw'][i, ...] = test_img[i]
        out_file['pred'][i, ...] = test_pred[i]
        out_file['paz'][i, ...] = test_paz[i]
        out_file['phase'][i, ...] = test_phase[i]
        out_file['nslice'][i, ...] = test_slice[i]
    out_file['pixel_size'][...] = px_dim

    out_file.close()

    logging.info('\n ---------- computing metrics -------------')
    df = compute_metrics_on_directories_raw(os.path.join(out_path, paz, 'pred.hdf5'))
    df.to_excel(os.path.join(out_path, paz, 'Excel_df.xlsx'))

    print('saving images...')
    pdf_path = os.path.join(out_path, paz, 'plt_imgs.pdf')
    data = h5py.File(os.path.join(out_path, paz, 'pred.hdf5'), 'r')
    figs = []
    color = [(255, 0, 0), (0, 255, 0), (255, 255, 0)]
    for i in range(len(data['img_raw'])):
        img = data['img_raw'][i]
        img_o = cv2.normalize(src=img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        img_pred = cv2.cvtColor(img_o, cv2.COLOR_GRAY2RGB)
        for struc in [1, 2, 3]:
            pred = data['pred'][i].astype(np.uint8)
            if struc == 1:
                pred[pred != 1] = 0
            elif struc == 2:
                pred[pred == 1] = 0
            elif struc == 3:
                pred[pred != 3] = 0
            pred[pred != 0] = 1
            contours_pred, _ = cv2.findContours(image=pred, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(image=img_pred, contours=contours_pred, contourIdx=-1, color=color[struc-1], thickness=1, lineType=cv2.LINE_AA)

        fig = plt.figure(figsize=(14, 14))
        ax1 = fig.add_subplot(121)
        ax1.set_axis_off()
        ax1.imshow(img_o, cmap='gray')

        ax2 = fig.add_subplot(122)
        ax2.set_axis_off()
        ax2.imshow(img_pred)
        ax1.title.set_text('Raw_img')
        ax2.title.set_text('Automated')
        txt = str(data['phase'][i] + '_' + data['nslice'][i])
        plt.text(0.1, 0.65, txt, transform=fig.transFigure, size=18)
        figs.append(fig)
        # plt.show()
    data.close()

    with PdfPages(pdf_path) as pdf:
        for fig in figs:
            pdf.savefig(fig)
            plt.close()

logging.info('Average time per volume: %f' % (total_time / total_volumes))
