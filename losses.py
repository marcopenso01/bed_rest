from tensorflow.keras import backend as K
import numpy as np
import tensorflow as tf

smooth = 1.


def average_dice_coef(y_true, y_pred):
    loss = 0
    label_length = y_pred.get_shape().as_list()[-1]
    print(label_length)
    for num_label in range(label_length):
        y_true_f = K.flatten(y_true[..., num_label])
        y_pred_f = K.flatten(y_pred[..., num_label])
        intersection = K.sum(y_true_f * y_pred_f)
        loss += (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return loss / label_length


def average_dice_coef_loss(y_true, y_pred):
    return -average_dice_coef(y_true, y_pred)
