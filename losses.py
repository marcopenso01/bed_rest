from tensorflow.keras import backend as K
import numpy as np
import tensorflow as tf

smooth = 1.


def average_dice_coef(y_true, y_pred):
    loss = 0
    label_length = y_pred.get_shape().as_list()[-1]
    #print(label_length)
    only_foreground = True  #Exclude label 0 from evaluation
    if only_foreground:
        for num_label in range(1, label_length):
            y_true_f = K.flatten(y_true[..., num_label])
            y_pred_f = K.flatten(y_pred[..., num_label])
            intersection = K.sum(y_true_f * y_pred_f)
            loss += (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    else:
        for num_label in range(label_length):
            y_true_f = K.flatten(y_true[..., num_label])
            y_pred_f = K.flatten(y_pred[..., num_label])
            intersection = K.sum(y_true_f * y_pred_f)
            loss += (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return loss / label_length


def average_dice_coef_loss(y_true, y_pred):
    dice_loss = 1 - average_dice_coef(y_true, y_pred)
    return dice_loss


def weighted_categorical_crossentropy(weights=[0.1, 0.3, 0.3, 0.3]):
    """
    A weighted version of keras.objectives.categorical_crossentropy

    weights: numpy array of shape (C,) where C is the number of classes
             np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
    """
    weights = K.variable(weights)

    def loss_function(y_true, y_pred):
        #cce = tf.keras.losses.CategoricalCrossentropy()

        axis = [1, 2]
        # clip to prevent NaN and Inf
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calculate cross entropy
        cross_entropy = -y_true * K.log(y_pred) * weights
        # sum over classes
        cross_entropy = K.mean(K.sum(cross_entropy, axis=[-1]))
        return cross_entropy

    return loss_function


def combo_loss(alpha=1.0, weights=[0.1, 0.3, 0.3, 0.3]):
    """Combo Loss:
        Parameters
        ----------
        alpha : float, optional
            controls weighting of dice and cross-entropy loss., by default 0.5
        """
    weights = K.variable(weights)

    def loss_function(y_true, y_pred):
        dice = average_dice_coef_loss(y_true, y_pred)

        #cross_entropy = weighted_categorical_crossentropy()(y_true, y_pred)

        axis = [1, 2]
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * K.log(y_pred) * weights
        cross_entropy = K.mean(K.sum(cross_entropy, axis[-1]))
        combo_loss = (alpha * cross_entropy) + (alpha * dice)

        return combo_loss

    return loss_function
