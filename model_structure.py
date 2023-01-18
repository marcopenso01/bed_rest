import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
import logging

logging.basicConfig(
    level=logging.INFO  # allow DEBUG level messages to pass through the logger
)


def activation_block(x):
    #x = custom_gelu(x)
    x = ReLU()(x)
    return BatchNormalization()(x)


def conv_mixer_block(x, filters: int, kernel_size: int):
    # Depthwise convolution.
    x0 = x
    x = DepthwiseConv2D(kernel_size=kernel_size, padding="same")(x)
    x = Add()([activation_block(x), x0])  # Residual.
    # Pointwise convolution.
    x = Conv2D(filters, kernel_size=1, kernel_initializer='he_normal')(x)
    x = activation_block(x)
    return x


def selective_kernel(input1, input2, channel, ratio=8):
    '''
    input1: input tensor from the 2D network (x,y,channels)
    input2: input tensor from the 3D network (x,y,depth,channels)
    channel: channel number of the result
    return: processed tensor
    '''
    inputs_shape = tf.shape(input1)
    b, h, w = inputs_shape[0], inputs_shape[1], inputs_shape[2]
    xs = []
    xs.append(input1)

    conv2 = ReLU()(BatchNormalization()(Conv3D(1, 1, padding='same', kernel_initializer='he_normal')(input2)))
    conv2 = tf.keras.backend.squeeze(conv2, axis=-1)
    conv2 = ReLU()(BatchNormalization()(Conv2D(channel, 3, padding='same', kernel_initializer='he_normal')(conv2)))
    xs.append(conv2)

    conv_unite = Add()(xs)

    avg_pool = GlobalAveragePooling2D()(conv_unite)
    avg_pool = Reshape((1, 1, channel))(avg_pool)
    # output_shape=[b, 1, 1, channel]

    z = ReLU()(
        BatchNormalization()(Conv2D(channel // ratio, 1, kernel_initializer='he_normal', padding='same')(avg_pool)))
    x = Conv2D(channel * 2, 1, kernel_initializer='he_normal', padding='same')(z)
    x = Reshape([1, 1, channel, 2])(x)
    scale = Softmax()(x)

    #xstack = Lambda(lambda x: tf.keras.backend.stack(x, axis=-1),
    #           output_shape=[b, h, w, channel, 2])(xs)
    xstack = tf.keras.backend.stack(xs, axis=-1)

    f = tf.multiply(scale, xstack, name='product')
    f = tf.reduce_sum(f, axis=-1, name='sum')

    return f[0:4]


def bi_fusion(input1, input2):
    '''
    input1: input tensor from the 2D network (x,y,channels)
    input2: input tensor from the 3D network (x,y,depth,channels)
    channel: channel number of the result
    return: processed tensor
    '''
    channel = input1.get_shape().as_list()[-1]
    conv2 = ReLU()(BatchNormalization()(Conv3D(1, 1, padding='same', kernel_initializer='he_normal')(input2)))
    conv2 = tf.keras.backend.squeeze(conv2, axis=-1)
    conv2 = ReLU()(BatchNormalization()(Conv2D(channel, 3, padding='same', kernel_initializer='he_normal')(conv2)))
    conv2 = squeeze_excite_block(conv2)

    conv1 = squeeze_excite_block(input1)

    return Add()([conv1, conv2])


def squeeze_excite_block(input_tensor, ratio=8):
    """ Create a channel-wise squeeze-excite block
        Args:
            input_tensor: input Keras tensor
            ratio: number of output filters
        Returns: a Keras tensor
        References
        -   [Squeeze and Excitation Networks](https://arxiv.org/abs/1709.01507)
    """
    init = input_tensor
    channel = input_tensor.get_shape().as_list()[-1]
    se = GlobalAveragePooling2D()(init)
    se = Reshape((1, 1, channel))(se)
    se = Dense(channel // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(channel, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    x = multiply([init, se])
    return x


def cbam_block(cbam_feature, ratio=8):
    """Contains the implementation of Convolutional Block Attention Module(CBAM) block.
    As described in https://arxiv.org/abs/1807.06521.
    """
    cbam_feature = channel_attention(cbam_feature, ratio)
    cbam_feature = spatial_attention(cbam_feature)
    return cbam_feature


def spatial_attention(input_feature):
    kernel_size = 7
    channel = input_feature.get_shape().as_list()[-1]

    avg_pool = tf.reduce_mean(input_feature, [1, 2], keepdims=True)

    max_pool = tf.reduce_max(input_feature, [1, 2], keepdims=True)

    concat = Concatenate(axis=-1)([avg_pool, max_pool])
    cbam_feature = Conv2D(filters=1,
                          kernel_size=kernel_size,
                          strides=1,
                          padding='same',
                          activation='sigmoid',
                          kernel_initializer='he_normal',
                          use_bias=False)(concat)

    return multiply([input_feature, cbam_feature])


def channel_attention(input_feature, ratio=8):
    channel = input_feature.get_shape().as_list()[-1]

    shared_layer_one = Dense(channel // ratio,
                             activation='relu',
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    shared_layer_two = Dense(channel,
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')

    avg_pool = GlobalAveragePooling2D()(input_feature)
    avg_pool = Reshape((1, 1, channel))(avg_pool)
    avg_pool = shared_layer_one(avg_pool)
    avg_pool = shared_layer_two(avg_pool)

    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1, 1, channel))(max_pool)
    max_pool = shared_layer_one(max_pool)
    max_pool = shared_layer_two(max_pool)

    cbam_feature = Add()([avg_pool, max_pool])
    cbam_feature = Activation('sigmoid')(cbam_feature)

    return multiply([input_feature, cbam_feature])



def DilatedSpatialConv(dspp_input, filters=32):
    out_1 = ReLU()(BatchNormalization()(Conv2D(filters, (3, 3), padding='same', kernel_initializer='he_normal',
                                               dilation_rate=1)(dspp_input)))
    out_4 = ReLU()(BatchNormalization()(Conv2D(filters, (3, 3), padding='same', kernel_initializer='he_normal',
                                               dilation_rate=4)(dspp_input)))
    out_7 = ReLU()(BatchNormalization()(Conv2D(filters, (3, 3), padding='same', kernel_initializer='he_normal',
                                               dilation_rate=7)(dspp_input)))

    x = Concatenate(axis=-1)([out_1, out_4, out_7])
    return x


def Unet(input_size1=(160, 160, 1), n_filt=32):
    input_model1 = Input(input_size1)

    # layer1 2D
    x1 = ReLU()(BatchNormalization()(Conv2D(n_filt, (3, 3), padding='same', kernel_initializer='he_normal')(input_model1)))
    conv1 = ReLU()(BatchNormalization()(Conv2D(n_filt, (3, 3), padding='same', kernel_initializer='he_normal')(x1)))
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    # layer2 2D
    conv2 = ReLU()(BatchNormalization()(Conv2D(n_filt * 2, (3, 3), padding='same', kernel_initializer='he_normal')(pool1)))
    conv2 = ReLU()(BatchNormalization()(Conv2D(n_filt * 2, (3, 3), padding='same', kernel_initializer='he_normal')(conv2)))
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # layer3 2D
    conv3 = ReLU()(BatchNormalization()(Conv2D(n_filt * 4, (3, 3), padding='same', kernel_initializer='he_normal')(pool2)))
    conv3 = ReLU()(BatchNormalization()(Conv2D(n_filt * 4, (3, 3), padding='same', kernel_initializer='he_normal')(conv3)))
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    # layer4 2D
    conv4 = ReLU()(BatchNormalization()(Conv2D(n_filt * 8, (3, 3), padding='same', kernel_initializer='he_normal')(pool3)))
    conv4 = ReLU()(BatchNormalization()(Conv2D(n_filt * 8, (3, 3), padding='same', kernel_initializer='he_normal')(conv4)))
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    # layer5 2D
    conv5 = ReLU()(BatchNormalization()(Conv2D(n_filt * 16, (3, 3), padding='same', kernel_initializer='he_normal')(pool4)))
    conv5 = ReLU()(BatchNormalization()(Conv2D(n_filt * 16, (3, 3), padding='same', kernel_initializer='he_normal')(conv5)))

    up4 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=3)

    conv6 = ReLU()(BatchNormalization()(Conv2D(n_filt * 8, (3, 3), padding='same', kernel_initializer='he_normal')(up4)))
    conv6 = ReLU()(BatchNormalization()(Conv2D(n_filt * 8, (3, 3), padding='same', kernel_initializer='he_normal')(conv6)))

    up3 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=3)

    conv7 = ReLU()(BatchNormalization()(Conv2D(n_filt * 4, (3, 3), padding='same', kernel_initializer='he_normal')(up3)))
    conv7 = ReLU()(BatchNormalization()(Conv2D(n_filt * 4, (3, 3), padding='same', kernel_initializer='he_normal')(conv7)))

    up2 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=3)

    conv8 = ReLU()(BatchNormalization()(Conv2D(n_filt * 2, (3, 3), padding='same', kernel_initializer='he_normal')(up2)))
    conv8 = ReLU()(BatchNormalization()(Conv2D(n_filt * 2, (3, 3), padding='same', kernel_initializer='he_normal')(conv8)))

    up1 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=3)

    conv9 = ReLU()(BatchNormalization()(Conv2D(n_filt, (3, 3), padding='same', kernel_initializer='he_normal')(up1)))
    conv9 = ReLU()(BatchNormalization()(Conv2D(n_filt, (3, 3), padding='same', kernel_initializer='he_normal')(conv9)))

    output = Conv2D(4, (1, 1), activation='softmax')(conv9)

    model = Model(inputs=input_model1, outputs=output)
    logging.info('Finish building model')

    return model


def ConvMixUnet(input_size1=(160, 160, 1), n_filt=32):
    input_model1 = Input(input_size1)

    # layer1 2D
    conv1 = ReLU()(BatchNormalization()(Conv2D(n_filt, (5, 5), padding='same', kernel_initializer='he_normal')(input_model1)))
    conv1 = conv_mixer_block(conv1, filters=n_filt, kernel_size=3)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    # layer2 2D
    conv2 = conv_mixer_block(pool1, filters=n_filt * 2, kernel_size=3)
    conv2 = conv_mixer_block(conv2, filters=n_filt * 2, kernel_size=3)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    # layer3 2D
    conv3 = conv_mixer_block(pool2, filters=n_filt * 4, kernel_size=3)
    conv3 = conv_mixer_block(conv3, filters=n_filt * 4, kernel_size=3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    # layer4 2D
    conv4 = conv_mixer_block(pool3, filters=n_filt * 8, kernel_size=3)
    conv4 = conv_mixer_block(conv4, filters=n_filt * 8, kernel_size=3)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    # layer5 2D
    conv5 = conv_mixer_block(pool4, filters=n_filt * 16, kernel_size=3)
    conv5 = conv_mixer_block(conv5, filters=n_filt * 16, kernel_size=3)

    up4 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=3)

    conv6 = conv_mixer_block(up4, filters=n_filt * 8, kernel_size=3)
    conv6 = conv_mixer_block(conv6, filters=n_filt * 8, kernel_size=3)

    up3 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=3)

    conv7 = conv_mixer_block(up3, filters=n_filt * 4, kernel_size=3)
    conv7 = conv_mixer_block(conv7, filters=n_filt * 4, kernel_size=3)

    up2 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=3)

    conv8 = conv_mixer_block(up2, filters=n_filt * 2, kernel_size=3)
    conv8 = conv_mixer_block(conv8, filters=n_filt * 2, kernel_size=3)

    up1 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=3)

    conv9 = conv_mixer_block(up1, filters=n_filt, kernel_size=3)
    conv9 = conv_mixer_block(conv9, filters=n_filt, kernel_size=3)

    output = Conv2D(4, (1, 1), activation='softmax')(conv9)

    model = Model(inputs=input_model1, outputs=output)
    logging.info('Finish building model')

    return model


def ConvMixSkipUnet(input_size1=(160, 160, 1), n_filt=32):
    input_model1 = Input(input_size1)

    # layer1 2D
    conv1 = ReLU()(BatchNormalization()(Conv2D(n_filt, (5, 5), padding='same', kernel_initializer='he_normal')(input_model1)))
    conv1 = conv_mixer_block(conv1, filters=n_filt, kernel_size=3)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    # layer2 2D
    conv2 = conv_mixer_block(pool1, filters=n_filt * 2, kernel_size=3)
    conv2 = conv_mixer_block(conv2, filters=n_filt * 2, kernel_size=3)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    # layer3 2D
    conv3 = conv_mixer_block(pool2, filters=n_filt * 4, kernel_size=3)
    conv3 = conv_mixer_block(conv3, filters=n_filt * 4, kernel_size=3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    # layer4 2D
    conv4 = conv_mixer_block(pool3, filters=n_filt * 8, kernel_size=3)
    conv4 = conv_mixer_block(conv4, filters=n_filt * 8, kernel_size=3)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    # layer5 2D
    conv5 = conv_mixer_block(pool4, filters=n_filt * 16, kernel_size=3)
    conv5 = conv_mixer_block(conv5, filters=n_filt * 16, kernel_size=3)

    up4 = UpSampling2D(size=(2, 2))(conv5)
    skip4_0 = conv_mixer_block(conv4, filters=n_filt, kernel_size=3)
    skip4_1 = concatenate([conv4, skip4_0], axis=3)
    conc4 = concatenate([up4, skip4_1], axis=3)

    conv6 = conv_mixer_block(conc4, filters=n_filt * 8, kernel_size=3)
    conv6 = conv_mixer_block(conv6, filters=n_filt * 8, kernel_size=3)

    up3 = UpSampling2D(size=(2, 2))(conv6)
    skip3_0 = conv_mixer_block(conv3, filters=n_filt, kernel_size=3)
    skip3_1 = concatenate([conv3, skip3_0], axis=3)
    skip3_2 = conv_mixer_block(skip3_1, filters=n_filt, kernel_size=3)
    skip3_3 = concatenate([skip3_1, skip3_2], axis=3)
    conc3 = concatenate([up3, skip3_3], axis=3)

    conv7 = conv_mixer_block(conc3, filters=n_filt * 4, kernel_size=3)
    conv7 = conv_mixer_block(conv7, filters=n_filt * 4, kernel_size=3)

    up2 = UpSampling2D(size=(2, 2))(conv7)
    skip2_0 = conv_mixer_block(conv2, filters=n_filt, kernel_size=3)
    skip2_1 = concatenate([conv2, skip2_0], axis=3)
    skip2_2 = conv_mixer_block(skip2_1, filters=n_filt, kernel_size=3)
    skip2_3 = concatenate([skip2_1, skip2_2], axis=3)
    skip2_4 = conv_mixer_block(skip2_3, filters=n_filt, kernel_size=3)
    skip2_5 = concatenate([skip2_3, skip2_4], axis=3)
    conc2 = concatenate([up2, skip2_5], axis=3)

    conv8 = conv_mixer_block(conc2, filters=n_filt * 2, kernel_size=3)
    conv8 = conv_mixer_block(conv8, filters=n_filt * 2, kernel_size=3)

    up1 = UpSampling2D(size=(2, 2))(conv8)
    skip1_0 = conv_mixer_block(conv1, filters=n_filt, kernel_size=3)
    skip1_1 = concatenate([conv1, skip1_0], axis=3)
    skip1_2 = conv_mixer_block(skip1_1, filters=n_filt, kernel_size=3)
    skip1_3 = concatenate([skip1_1, skip1_2], axis=3)
    skip1_4 = conv_mixer_block(skip1_3, filters=n_filt, kernel_size=3)
    skip1_5 = concatenate([skip1_3, skip1_4], axis=3)
    skip1_6 = conv_mixer_block(skip1_5, filters=n_filt, kernel_size=3)
    skip1_7 = concatenate([skip1_5, skip1_6], axis=3)
    conc1 = concatenate([up1, skip1_7], axis=3)

    conv9 = conv_mixer_block(conc1, filters=n_filt, kernel_size=3)
    conv9 = conv_mixer_block(conv9, filters=n_filt, kernel_size=3)

    output = Conv2D(4, (1, 1), activation='softmax')(conv9)

    model = Model(inputs=input_model1, outputs=output)
    logging.info('Finish building model')

    return model


def Unet3d(input_size1=(160, 160, 1), input_size2=(160, 160, 1),
         input_size3=(160, 160, 1), n_filt=32):

    input_model1 = Input(input_size1)
    input_model2 = Input(input_size2)
    input_model3 = Input(input_size3)

    # layer1 2D
    x1 = ReLU()(
        BatchNormalization()(Conv2D(n_filt, (5, 5), padding='same', kernel_initializer='he_normal')(input_model1)))
    conv1 = conv_mixer_block(x1, filters=n_filt, kernel_size=3)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    # layer1 3D
    input_model3d = Concatenate(axis=-1)([input_model2, input_model1, input_model3])
    input_model3d = tf.expand_dims(input_model3d, -1)
    x1_2 = ReLU()(
        BatchNormalization()(Conv3D(n_filt, 3, padding='same', kernel_initializer='he_normal')(input_model3d)))
    conv1_2 = ReLU()(BatchNormalization()(Conv3D(n_filt, 3, padding='same', kernel_initializer='he_normal')(x1_2)))
    pool1_2 = MaxPooling3D(pool_size=(2, 2, 1))(conv1_2)

    # layer2 2D
    conv2 = conv_mixer_block(pool1, filters=n_filt * 2, kernel_size=3)
    conv2 = conv_mixer_block(conv2, filters=n_filt * 2, kernel_size=3)
    # layer2 3D
    conv2_2 = ReLU()(
        BatchNormalization()(Conv3D(n_filt * 2, 3, padding='same', kernel_initializer='he_normal')(pool1_2)))
    conv2_2 = ReLU()(
        BatchNormalization()(Conv3D(n_filt * 2, 3, padding='same', kernel_initializer='he_normal')(conv2_2)))
    pool2_2 = MaxPooling3D(pool_size=(2, 2, 1))(conv2_2)

    select1 = selective_kernel(conv2, conv2_2, n_filt * 2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(select1)

    # layer3 2D
    conv3 = conv_mixer_block(pool2, filters=n_filt * 4, kernel_size=3)
    conv3 = conv_mixer_block(conv3, filters=n_filt * 4, kernel_size=3)
    # layer3 3D
    conv3_2 = ReLU()(
        BatchNormalization()(Conv3D(n_filt * 4, 3, padding='same', kernel_initializer='he_normal')(pool2_2)))
    conv3_2 = ReLU()(
        BatchNormalization()(Conv3D(n_filt * 4, 3, padding='same', kernel_initializer='he_normal')(conv3_2)))

    select2 = selective_kernel(conv3, conv3_2, n_filt * 4)
    pool3 = MaxPooling2D(pool_size=(2, 2))(select2)

    # layer4 2D
    conv4 = conv_mixer_block(pool3, filters=n_filt * 8, kernel_size=3)
    conv4 = conv_mixer_block(conv4, filters=n_filt * 8, kernel_size=3)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    # layer5 2D
    conv5 = conv_mixer_block(pool4, filters=n_filt * 16, kernel_size=3)
    conv5 = conv_mixer_block(conv5, filters=n_filt * 16, kernel_size=3)

    up4 = UpSampling2D(size=(2, 2))(conv5)
    skip4_0 = conv_mixer_block(conv4, filters=n_filt, kernel_size=3)
    skip4_1 = concatenate([conv4, skip4_0], axis=3)
    conc4 = concatenate([up4, skip4_1], axis=3)

    conv6 = conv_mixer_block(conc4, filters=n_filt * 8, kernel_size=3)
    conv6 = conv_mixer_block(conv6, filters=n_filt * 8, kernel_size=3)

    up3 = UpSampling2D(size=(2, 2))(conv6)
    skip3_0 = conv_mixer_block(conv3, filters=n_filt, kernel_size=3)
    skip3_1 = concatenate([conv3, skip3_0], axis=3)
    skip3_2 = conv_mixer_block(skip3_1, filters=n_filt, kernel_size=3)
    skip3_3 = concatenate([skip3_1, skip3_2], axis=3)
    conc3 = concatenate([up3, skip3_3], axis=3)

    conv7 = conv_mixer_block(conc3, filters=n_filt * 4, kernel_size=3)
    conv7 = conv_mixer_block(conv7, filters=n_filt * 4, kernel_size=3)

    up2 = UpSampling2D(size=(2, 2))(conv7)
    skip2_0 = conv_mixer_block(conv2, filters=n_filt, kernel_size=3)
    skip2_1 = concatenate([conv2, skip2_0], axis=3)
    skip2_2 = conv_mixer_block(skip2_1, filters=n_filt, kernel_size=3)
    skip2_3 = concatenate([skip2_1, skip2_2], axis=3)
    skip2_4 = conv_mixer_block(skip2_3, filters=n_filt, kernel_size=3)
    skip2_5 = concatenate([skip2_3, skip2_4], axis=3)
    conc2 = concatenate([up2, skip2_5], axis=3)

    conv8 = conv_mixer_block(conc2, filters=n_filt * 2, kernel_size=3)
    conv8 = conv_mixer_block(conv8, filters=n_filt * 2, kernel_size=3)

    up1 = UpSampling2D(size=(2, 2))(conv8)
    skip1_0 = conv_mixer_block(conv1, filters=n_filt, kernel_size=3)
    skip1_1 = concatenate([conv1, skip1_0], axis=3)
    skip1_2 = conv_mixer_block(skip1_1, filters=n_filt, kernel_size=3)
    skip1_3 = concatenate([skip1_1, skip1_2], axis=3)
    skip1_4 = conv_mixer_block(skip1_3, filters=n_filt, kernel_size=3)
    skip1_5 = concatenate([skip1_3, skip1_4], axis=3)
    skip1_6 = conv_mixer_block(skip1_5, filters=n_filt, kernel_size=3)
    skip1_7 = concatenate([skip1_5, skip1_6], axis=3)
    conc1 = concatenate([up1, skip1_7], axis=3)

    conv9 = conv_mixer_block(conc1, filters=n_filt, kernel_size=3)
    conv9 = conv_mixer_block(conv9, filters=n_filt, kernel_size=3)

    conv_out = Conv2D(4, 1, activation='softmax', padding='same', kernel_initializer='he_normal')(conv9)

    model = Model(inputs=[input_model1, input_model2, input_model3], outputs=conv_out)
    logging.info('Finish building model')
    return model


def AttUnet3d(input_size1=(160, 160, 1), input_size2=(160, 160, 1),
         input_size3=(160, 160, 1), n_filt=32):

    input_model1 = Input(input_size1)
    input_model2 = Input(input_size2)
    input_model3 = Input(input_size3)

    # layer1 2D
    x1 = DilatedSpatialConv(input_model1, filters=n_filt)
    conv1 = conv_mixer_block(x1, filters=n_filt, kernel_size=3)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    # layer1 3D
    input_model3d = Concatenate(axis=-1)([input_model2, input_model1, input_model3])
    input_model3d = tf.expand_dims(input_model3d, -1)
    x1_2 = ReLU()(
        BatchNormalization()(Conv3D(n_filt, 3, padding='same', kernel_initializer='he_normal')(input_model3d)))
    conv1_2 = ReLU()(BatchNormalization()(Conv3D(n_filt, 3, padding='same', kernel_initializer='he_normal')(x1_2)))
    pool1_2 = MaxPooling3D(pool_size=(2, 2, 1))(conv1_2)

    # layer2 2D
    conv2 = conv_mixer_block(pool1, filters=n_filt * 2, kernel_size=3)
    conv2 = conv_mixer_block(conv2, filters=n_filt * 2, kernel_size=3)
    # layer2 3D
    conv2_2 = ReLU()(
        BatchNormalization()(Conv3D(n_filt * 2, 3, padding='same', kernel_initializer='he_normal')(pool1_2)))
    conv2_2 = ReLU()(
        BatchNormalization()(Conv3D(n_filt * 2, 3, padding='same', kernel_initializer='he_normal')(conv2_2)))
    pool2_2 = MaxPooling3D(pool_size=(2, 2, 1))(conv2_2)

    select1 = bi_fusion(conv2, conv2_2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(select1)

    # layer3 2D
    conv3 = conv_mixer_block(pool2, filters=n_filt * 4, kernel_size=3)
    conv3 = conv_mixer_block(conv3, filters=n_filt * 4, kernel_size=3)
    # layer3 3D
    conv3_2 = ReLU()(
        BatchNormalization()(Conv3D(n_filt * 4, 3, padding='same', kernel_initializer='he_normal')(pool2_2)))
    conv3_2 = ReLU()(
        BatchNormalization()(Conv3D(n_filt * 4, 3, padding='same', kernel_initializer='he_normal')(conv3_2)))

    select2 = bi_fusion(conv3, conv3_2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(select2)

    # layer4 2D
    conv4 = conv_mixer_block(pool3, filters=n_filt * 8, kernel_size=3)
    conv4 = conv_mixer_block(conv4, filters=n_filt * 8, kernel_size=3)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    # layer5 2D
    conv5 = conv_mixer_block(pool4, filters=n_filt * 16, kernel_size=3)
    conv5 = spatial_attention(conv5)
    conv5 = conv_mixer_block(conv5, filters=n_filt * 16, kernel_size=3)

    up4 = UpSampling2D(size=(2, 2))(conv5)
    skip4 = cbam_block(conv4)
    conc4 = concatenate([up4, skip4], axis=3)

    conv6 = conv_mixer_block(conc4, filters=n_filt * 8, kernel_size=3)
    conv6 = conv_mixer_block(conv6, filters=n_filt * 8, kernel_size=3)

    up3 = UpSampling2D(size=(2, 2))(conv6)
    skip3 = cbam_block(conv3)
    conc3 = concatenate([up3, skip3], axis=3)

    conv7 = conv_mixer_block(conc3, filters=n_filt * 4, kernel_size=3)
    conv7 = conv_mixer_block(conv7, filters=n_filt * 4, kernel_size=3)

    up2 = UpSampling2D(size=(2, 2))(conv7)
    skip2 = cbam_block(conv2)
    conc2 = concatenate([up2, skip2], axis=3)

    conv8 = conv_mixer_block(conc2, filters=n_filt * 2, kernel_size=3)
    conv8 = conv_mixer_block(conv8, filters=n_filt * 2, kernel_size=3)

    up1 = UpSampling2D(size=(2, 2))(conv8)
    skip1 = cbam_block(conv1)
    conc1 = concatenate([up1, skip1], axis=3)

    conv9 = conv_mixer_block(conc1, filters=n_filt, kernel_size=3)
    conv9 = conv_mixer_block(conv9, filters=n_filt, kernel_size=3)

    conv_out = Conv2D(4, 1, activation='softmax', padding='same', kernel_initializer='he_normal')(conv9)

    model = Model(inputs=[input_model1, input_model2, input_model3], outputs=conv_out)
    logging.info('Finish building model')
    return model


def ConvMixer(input_size1=(160, 160, 1), input_size2=(160, 160, 1), input_size3=(160, 160, 1), n_filt=32):

    input_model1 = Input(input_size1)
    input_model2 = Input(input_size2)
    input_model3 = Input(input_size3)

    input_model3d = Concatenate(axis=-1)([input_model2, input_model1, input_model3])

    # layer1
    conv1 = ReLU()(
        BatchNormalization()(Conv2D(n_filt, (5, 5), padding='same', kernel_initializer='he_normal')(input_model1)))
    conv1 = conv_mixer_block(conv1, filters=n_filt, kernel_size=3)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    # layer2
    x2 = conv_stem(input_model3d, filters=256, patch_size=2)
    for _ in range(10):
        x2 = conv_mixer_block(x2, filters=256, kernel_size=5)

    conc2 = concatenate([pool1, x2], axis=3)
    conv2 = conv_mixer_block(conc2, filters=n_filt * 2, kernel_size=3)
    conv2 = conv_mixer_block(conv2, filters=n_filt * 2, kernel_size=3)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # layer3
    x3 = conv_stem(input_model3d, filters=256, patch_size=4)
    for _ in range(10):
        x3 = conv_mixer_block(x3, filters=256, kernel_size=5)

    conc3 = concatenate([pool2, x3], axis=3)
    conv3 = conv_mixer_block(conc3, filters=n_filt * 4, kernel_size=3)
    conv3 = conv_mixer_block(conv3, filters=n_filt * 4, kernel_size=3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    # layer4
    conv4 = conv_mixer_block(pool3, filters=n_filt * 8, kernel_size=3)
    conv4 = conv_mixer_block(conv4, filters=n_filt * 8, kernel_size=3)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    # layer5
    conv5 = conv_mixer_block(pool4, filters=n_filt * 16, kernel_size=3)
    conv5 = spatial_attention(conv5)
    conv5 = conv_mixer_block(conv5, filters=n_filt * 16, kernel_size=3)

    up4 = UpSampling2D(size=(2, 2))(conv5)
    skip4 = ResPath(conv4, length=1)
    conc6 = concatenate([up4, skip4], axis=3)
    conv6 = conv_mixer_block(conc6, filters=n_filt * 8, kernel_size=3)
    conv6 = conv_mixer_block(conv6, filters=n_filt * 8, kernel_size=3)

    up3 = UpSampling2D(size=(2, 2))(conv6)
    skip3 = ResPath(conv3, length=2)
    conc7 = concatenate([up3, skip3], axis=3)
    conv7 = conv_mixer_block(conc7, filters=n_filt * 4, kernel_size=3)
    conv7 = conv_mixer_block(conv7, filters=n_filt * 4, kernel_size=3)

    up2 = UpSampling2D(size=(2, 2))(conv7)
    skip2 = ResPath(conv2, length=3)
    conc8 = concatenate([up2, skip2], axis=3)
    conv8 = conv_mixer_block(conc8, filters=n_filt * 2, kernel_size=3)
    conv8 = conv_mixer_block(conv8, filters=n_filt * 2, kernel_size=3)

    up1 = UpSampling2D(size=(2, 2))(conv8)
    skip1 = ResPath(conv1, length=4)
    conc9 = concatenate([up1, skip1], axis=3)
    conv9 = conv_mixer_block(conc9, filters=n_filt, kernel_size=3)
    conv9 = conv_mixer_block(conv9, filters=n_filt, kernel_size=3)

    conv_out = Conv2D(4, 1, activation='softmax', padding='same', kernel_initializer='he_normal')(conv9)

    model = Model(inputs=[input_model1, input_model2, input_model3], outputs=conv_out)
    logging.info('Finish building model')
    return model


def ConvMixer2(input_size1=(160, 160, 1), n_filt=32):

    input_model1 = Input(input_size1)

    #x1 = conv_stem(input_model1, filters=256, patch_size=1)
    #for _ in range(10):
    #    x1 = conv_mixer_block(x1, filters=256, kernel_size=10)

    # layer1
    conv1 = ReLU()(
        BatchNormalization()(Conv2D(n_filt, (5, 5), padding='same', kernel_initializer='he_normal')(input_model1)))
    conv1 = conv_mixer_block(conv1, filters=n_filt, kernel_size=3)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    # layer2
    x2 = conv_stem(input_model1, filters=256, patch_size=2)
    for _ in range(10):
        x2 = conv_mixer_block(x2, filters=256, kernel_size=10)

    conc2 = concatenate([pool1, x2], axis=3)
    conv2 = conv_mixer_block(conc2, filters=n_filt * 2, kernel_size=3)
    conv2 = conv_mixer_block(conv2, filters=n_filt * 2, kernel_size=3)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # layer3
    x3 = conv_stem(input_model1, filters=256, patch_size=4)
    for _ in range(10):
        x3 = conv_mixer_block(x3, filters=256, kernel_size=10)

    conc3 = concatenate([pool2, x3], axis=3)
    conv3 = conv_mixer_block(conc3, filters=n_filt * 4, kernel_size=3)
    conv3 = conv_mixer_block(conv3, filters=n_filt * 4, kernel_size=3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    # layer4
    conv4 = conv_mixer_block(pool3, filters=n_filt * 8, kernel_size=3)
    conv4 = conv_mixer_block(conv4, filters=n_filt * 8, kernel_size=3)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    # layer5
    conv5 = conv_mixer_block(pool4, filters=n_filt * 16, kernel_size=3)
    conv5 = spatial_attention(conv5)
    conv5 = conv_mixer_block(conv5, filters=n_filt * 16, kernel_size=3)

    up4 = UpSampling2D(size=(2, 2))(conv5)
    skip4 = ResPath(conv4, length=1)
    conc6 = concatenate([up4, skip4], axis=3)
    conv6 = conv_mixer_block(conc6, filters=n_filt * 8, kernel_size=3)
    conv6 = conv_mixer_block(conv6, filters=n_filt * 8, kernel_size=3)

    up3 = UpSampling2D(size=(2, 2))(conv6)
    skip3 = ResPath(conv3, length=2)
    conc7 = concatenate([up3, skip3], axis=3)
    conv7 = conv_mixer_block(conc7, filters=n_filt * 4, kernel_size=3)
    conv7 = conv_mixer_block(conv7, filters=n_filt * 4, kernel_size=3)

    up2 = UpSampling2D(size=(2, 2))(conv7)
    skip2 = ResPath(conv2, length=3)
    conc8 = concatenate([up2, skip2], axis=3)
    conv8 = conv_mixer_block(conc8, filters=n_filt * 2, kernel_size=3)
    conv8 = conv_mixer_block(conv8, filters=n_filt * 2, kernel_size=3)

    up1 = UpSampling2D(size=(2, 2))(conv8)
    skip1 = ResPath(conv1, length=4)
    conc9 = concatenate([up1, skip1], axis=3)
    conv9 = ReLU()(
        BatchNormalization()(Conv2D(n_filt, (3, 3), padding='same', kernel_initializer='he_normal')(conc9)))
    conv9 = ReLU()(
        BatchNormalization()(Conv2D(n_filt, (3, 3), padding='same', kernel_initializer='he_normal')(conv9)))

    conv_out = Conv2D(4, 1, activation='softmax', padding='same', kernel_initializer='he_normal')(conv9)

    model = Model(inputs=input_model1, outputs=conv_out)
    logging.info('Finish building model')
    return model
