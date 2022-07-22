import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
import logging

logging.basicConfig(
    level=logging.INFO  # allow DEBUG level messages to pass through the logger
)

def Unet(input_size1=(160, 160, 1), input_size2=(160, 160, 1),
          input_size3=(160, 160, 1), n_filt=32):
    input_model1 = Input(input_size1)
    input_model2 = Input(input_size2)
    input_model3 = Input(input_size3)

    # layer1 2D
    x1 = ReLU()(BatchNormalization()(Conv2D(n_filt, 3, padding='same', kernel_initializer='he_normal')(input_model1)))
    conv1 = ReLU()(BatchNormalization()(Conv2D(n_filt, 3, padding='same', kernel_initializer='he_normal')(x1)))
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    # layer1 3D
    input_model3d = Concatenate(axis=-1)([input_model2, input_model1, input_model3])
    input_model3d = tf.expand_dims(input_model3d, -1)
    x1_2 = ReLU()(
        BatchNormalization()(Conv3D(n_filt, 3, padding='same', kernel_initializer='he_normal')(input_model3d)))
    conv1_2 = ReLU()(BatchNormalization()(Conv3D(n_filt, 3, padding='same', kernel_initializer='he_normal')(x1_2)))
    pool1_2 = MaxPooling3D(pool_size=(2, 2, 1))(conv1_2)
    # layer2 2D
    conv2 = ReLU()(BatchNormalization()(Conv2D(n_filt * 2, 3, padding='same', kernel_initializer='he_normal')(pool1)))
    conv2 = ReLU()(BatchNormalization()(Conv2D(n_filt * 2, 3, padding='same', kernel_initializer='he_normal')(conv2)))
    # layer2 3D
    conv2_2 = ReLU()(
        BatchNormalization()(Conv3D(n_filt * 2, 3, padding='same', kernel_initializer='he_normal')(pool1_2)))
    conv2_2 = ReLU()(
        BatchNormalization()(Conv3D(n_filt * 2, 3, padding='same', kernel_initializer='he_normal')(conv2_2)))
    pool2_2 = MaxPooling3D(pool_size=(2, 2, 1))(conv2_2)

    select1 = selective_kernel(conv2, conv2_2, n_filt * 2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(select1)

    # layer3 2D
    conv3 = ReLU()(BatchNormalization()(Conv2D(n_filt * 4, 3, padding='same', kernel_initializer='he_normal')(pool2)))
    conv3 = ReLU()(BatchNormalization()(Conv2D(n_filt * 4, 3, padding='same', kernel_initializer='he_normal')(conv3)))
    # layer3 3D
    conv3_2 = ReLU()(
        BatchNormalization()(Conv3D(n_filt * 4, 3, padding='same', kernel_initializer='he_normal')(pool2_2)))
    conv3_2 = ReLU()(
        BatchNormalization()(Conv3D(n_filt * 4, 3, padding='same', kernel_initializer='he_normal')(conv3_2)))

    select2 = selective_kernel(conv3, conv3_2, n_filt * 4)
    pool3 = MaxPooling2D(pool_size=(2, 2))(select2)

    # layer4 2D
    conv4 = ReLU()(BatchNormalization()(Conv2D(n_filt * 8, 3, padding='same', kernel_initializer='he_normal')(pool3)))
    conv4 = ReLU()(BatchNormalization()(Conv2D(n_filt * 8, 3, padding='same', kernel_initializer='he_normal')(conv4)))
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    # layer5 2D
    conv5 = ReLU()(BatchNormalization()(Conv2D(n_filt * 16, 3, padding='same', kernel_initializer='he_normal')(pool4)))
    conv5 = ReLU()(BatchNormalization()(Conv2D(n_filt * 16, 3, padding='same', kernel_initializer='he_normal')(conv5)))

    conv_up5 = ReLU()(BatchNormalization()(
        Conv2DTranspose(num_class, 4, strides=(2, 2), padding='same', activation='relu',
                        kernel_initializer='he_normal')(conv5)))

    merge6 = concatenate([conv_up5, conv4], axis=3)
    conv6 = ReLU()(BatchNormalization()(Conv2D(n_filt * 8, 3, padding='same', kernel_initializer='he_normal')(merge6)))
    conv6 = ReLU()(BatchNormalization()(Conv2D(n_filt * 8, 3, padding='same', kernel_initializer='he_normal')(conv6)))

    conv_up6 = ReLU()(BatchNormalization()(
        Conv2DTranspose(num_class, 4, strides=(2, 2), padding='same', activation='relu',
                        kernel_initializer='he_normal')(conv6)))

    merge7 = concatenate([conv_up6, conv3], axis=3)
    conv7 = ReLU()(BatchNormalization()(Conv2D(n_filt * 4, 3, padding='same', kernel_initializer='he_normal')(merge7)))
    conv7 = ReLU()(BatchNormalization()(Conv2D(n_filt * 4, 3, padding='same', kernel_initializer='he_normal')(conv7)))

    conv_up7 = ReLU()(BatchNormalization()(
        Conv2DTranspose(num_class, 4, strides=(2, 2), padding='same', activation='relu',
                        kernel_initializer='he_normal')(conv7)))

    merge8 = concatenate([conv_up7, conv2], axis=3)
    conv8 = ReLU()(BatchNormalization()(Conv2D(n_filt * 2, 3, padding='same', kernel_initializer='he_normal')(merge8)))
    conv8 = ReLU()(BatchNormalization()(Conv2D(n_filt * 2, 3, padding='same', kernel_initializer='he_normal')(conv8)))

    conv_up8 = ReLU()(BatchNormalization()(
        Conv2DTranspose(num_class, 4, strides=(2, 2), padding='same', activation='relu',
                        kernel_initializer='he_normal')(conv8)))

    merge9 = concatenate([conv_up8, conv1], axis=3)
    conv9 = ReLU()(BatchNormalization()(Conv2D(n_filt, 3, padding='same', kernel_initializer='he_normal')(merge9)))
    conv9 = ReLU()(BatchNormalization()(Conv2D(n_filt, 3, padding='same', kernel_initializer='he_normal')(conv9)))
    
    conv_out = Conv2D(4, 1, activation='softmax', padding='same', kernel_initializer='he_normal')(conv9)
    model = Model(inputs=[input_model1, input_model2, input_model3], outputs=conv_out)
    return model
