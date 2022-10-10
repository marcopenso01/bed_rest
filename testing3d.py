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
from tensorflow.python.client import device_lib

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


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
PATH
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
log_root = 'D:\BED_REST\logdir'
experiment_name = 'model6'
test_data_path = 'G:\BED-REST\data_test'
gt_exists = 'True'

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
TESTING AND EVALUATING THE MODEL
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
print('-' * 50)
print('Testing...')
print('-' * 50)
out_path = os.path.join(log_root, experiment_name, 'predictions')
if not tf.io.gfile.exists(out_path):
    tf.io.gfile.makedirs(out_path)

logging.info('\n------- loading model -----------')
model = tf.keras.models.load_model(os.path.join(log_root, experiment_name, 'model_weights.h5'),
                                   custom_objects={'loss_function': losses.combo_loss(), 'average_dice_coef': losses.average_dice_coef})

RAW = []
PRED = []
PAZ = []
MASK = []
PHS = []
total_time = 0
total_volumes = 0

for paz in os.listdir(test_data_path):

    start_time = time.time()

    logging.info(' --------------------------------------------')
    logging.info('------- Analysing paz: %s' % paz)
    logging.info(' --------------------------------------------')

    data = h5py.File(os.path.join(test_data_path, paz, 'pre_processing', 'pre_proc.hdf5'), 'r')
    test_img = data['img_raw'][()].astype('float32')
    test_label = data['mask'][()]
    test_right = data['img_right'][()].astype('float32')
    test_left = data['img_left'][()].astype('float32')
    test_up = data['img_up'][()].astype('float32')
    test_down = data['img_down'][()].astype('float32')
    test_paz = data['paz'][()]
    test_phase = data['phase'][()]
    #test_px = data['pixel_size'][()]
    data.close()

    for ii in range(len(test_img)):
        img = test_img[ii]
        img_left = test_left[ii]
        img_right = test_right[ii]
        RAW.append(img)
        PAZ.append(test_paz[ii])
        if test_phase[ii] == 'phase0':
            PHS.append('ED')
        else:
            PHS.append('ES')
        #PHS.append(test_phase[ii])
        if gt_exists:
            MASK.append(test_label[ii])

        img = normalize_image(img)
        img_left = normalize_image(img_left)
        img_right = normalize_image(img_right)
        img = np.float32(img)
        img_left = np.float32(img_left)
        img_right = np.float32(img_right)
        x = np.reshape(img, (1, img.shape[0], img.shape[1], 1))
        x1 = np.reshape(img_left, (1, img.shape[0], img.shape[1], 1))
        x2 = np.reshape(img_right, (1, img.shape[0], img.shape[1], 1))
        mask_out = model.predict((x, x1, x2))
        mask_out = np.squeeze(mask_out)
        mask_out = np.argmax(mask_out, axis=-1)
        PRED.append(mask_out)
    elapsed_time = time.time() - start_time
    total_time += elapsed_time
    total_volumes += 1
    logging.info('Evaluation of volume took %f secs.' % elapsed_time)


n_file = len(PRED)
out_file = h5py.File(os.path.join(out_path, 'pred.hdf5'), "w")

dt = h5py.special_dtype(vlen=str)
out_file.create_dataset('img_raw', [n_file] + [160, 160], dtype=np.float32)
out_file.create_dataset('pred', [n_file] + [160, 160], dtype=np.uint8)
#out_file.create_dataset('pixel_size', (n_file, 3), dtype=dt)
out_file.create_dataset('paz', (n_file,), dtype=dt)
out_file.create_dataset('phase', (n_file,), dtype=dt)
if gt_exists:
    out_file.create_dataset('mask', [n_file] + [160, 160], dtype=np.uint8)

for i in range(n_file):
    out_file['img_raw'][i, ...] = RAW[i]
    out_file['pred'][i, ...] = PRED[i]
    out_file['paz'][i, ...] = PAZ[i]
    out_file['phase'][i, ...] = PHS[i]
    #out_file['pixel_size'][i, ...] = PIXEL[i]
    if gt_exists:
        out_file['mask'][i, ...] = MASK[i]

out_file.close()

logging.info('Average time per volume: %f' % (total_time / total_volumes))

if gt_exists:
    logging.info('\n ---------- computing metrics -------------')
    metrics.main(out_path)
