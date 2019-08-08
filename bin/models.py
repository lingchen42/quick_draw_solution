import h5py
import numpy as np
from keras.models import Model, load_model, save_model
from keras.layers import Input, Dropout, BatchNormalization, Activation, ZeroPadding2D
from keras.layers import Add, AveragePooling2D, Flatten, Dense, UpSampling2D, GlobalAveragePooling2D
from keras.layers import GRU, Bidirectional, Input, Dense, LSTM
from keras.models import Model
from keras.layers.core import Lambda
from keras.layers.convolutional import Conv2D, Conv1D
from keras.layers.pooling import MaxPooling2D
from keras import backend as K
from keras import optimizers
import tensorflow as tf
import keras.applications as app 


np.random.seed(seed=1)
tf.set_random_seed(seed=1)
NCATS = 340


def BatchActivate(x):
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


def convolution_block(x, filters, size, strides=(1,1), padding='same', activation=True):
    x = Conv2D(filters, size, strides=strides, padding=padding)(x)
    if activation==True: x = BatchActivate(x)
    return x


def residual_block(blockInput, num_filters=16, batch_activate=False, add_idx=None, prefix=None):
    x = BatchActivate(blockInput)
    x = convolution_block(x, num_filters, (3,3))
    x = convolution_block(x, num_filters, (3,3), activation=False)
    if add_idx:
        x = Add(name="%s_add_%s"%(prefix, add_idx))([x, blockInput])
    else:
        x = Add()([x, blockInput])
    if batch_activate: x = BatchActivate(x)
    return x


def resnet_img256(size, DropoutRatio=0.25):
    input_layer = Input((size, size, 3))

    # 256 -> 128
    conv1 = Conv2D(64, (7,7), strides=(2, 2), activation=None, kernel_initializer='he_normal', padding='valid')(input_layer)
    conv1 = BatchNormalization(name='bn_conv1')(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = ZeroPadding2D(padding=(1, 1), name='pool1_pad')(conv1)
    conv1 = MaxPooling2D((3, 3), strides=(2, 2))(conv1)

    # 1st conv block
    conv1 = Conv2D(64, (3,3), activation=None, padding='same')(input_layer)
    conv1 = residual_block(conv1, 64)
    conv1 = residual_block(conv1, 64, True)
    pool1 = MaxPooling2D((2,2))(conv1)

    # 128 -> 64
    conv2 = Conv2D(128, (3,3), activation=None, padding='same')(pool1)
    conv2 = residual_block(conv2, 128)
    conv2 = residual_block(conv2, 128, True)
    pool2 = MaxPooling2D((2,2))(conv2)
    pool2 = Dropout(DropoutRatio)(pool2)

    # 64 -> 32
    conv3 = Conv2D(256, (3,3), activation=None, padding='same')(pool2)
    conv3 = residual_block(conv3, 256)
    conv3 = residual_block(conv3, 256, True)
    pool3 = MaxPooling2D((2,2))(conv3)
    pool3 = Dropout(DropoutRatio)(pool3)

    # 32 -> 16
    conv4 = Conv2D(512, (3,3), activation=None, padding='same')(pool3)
    conv4 = residual_block(conv4, 512)
    conv4 = residual_block(conv4, 512, True)
    pool4 = MaxPooling2D((2,2))(conv4)
    pool4 = Dropout(DropoutRatio)(pool4)

    # 16 -> 8
    conv5 = Conv2D(1024, (3,3), activation=None, padding='same')(pool4)
    conv5 = residual_block(conv5, 1024)
    conv5 = residual_block(conv5, 1024, True)
#     pool5 = MaxPooling2D((2,2))(conv5)
#     pool5 = Dropout(DropoutRatio)(pool5)

#     # 8 -> 4
#     conv6 = Conv2D(start_neurons*32, (3,3), activation=None, padding='same')(pool4)
#     conv6 = residual_block(conv5, start_neurons*32)
#     conv6 = residual_block(conv5, start_neurons*32, True)

    fc1 = GlobalAveragePooling2D(name='avg_pool')(conv5)
    output_layer = Dense(NCATS, activation='softmax', name='output')(fc1)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model


def resnet(size, start_neurons=32, DropoutRatio=0.25):

    input_layer = Input((size, size, 1))

    # 128 -> 64
    conv1 = Conv2D(start_neurons*1, (3,3), activation=None, padding='same')(input_layer)
    conv1 = residual_block(conv1, start_neurons*1)
    conv1 = residual_block(conv1, start_neurons*1, True)
    pool1 = MaxPooling2D((2,2))(conv1)

    # 64 -> 32
    conv2 = Conv2D(start_neurons*2, (3,3), activation=None, padding='same')(pool1)
    conv2 = residual_block(conv2, start_neurons*2)
    conv2 = residual_block(conv2, start_neurons*2, True)
    pool2 = MaxPooling2D((2,2))(conv2)
    pool2 = Dropout(DropoutRatio)(pool2)

    # 32 -> 16
    conv3 = Conv2D(start_neurons*4, (3,3), activation=None, padding='same')(pool2)
    conv3 = residual_block(conv3, start_neurons*4)
    conv3 = residual_block(conv3, start_neurons*4, True)
    pool3 = MaxPooling2D((2,2))(conv3)
    pool3 = Dropout(DropoutRatio)(pool3)

    # 16 -> 8
    conv4 = Conv2D(start_neurons*8, (3,3), activation=None, padding='same')(pool3)
    conv4 = residual_block(conv4, start_neurons*8)
    conv4 = residual_block(conv4, start_neurons*8, True)
    pool4 = MaxPooling2D((2,2))(conv4)
    pool4 = Dropout(DropoutRatio)(pool4)

    # 8 -> 8
    conv5 = Conv2D(start_neurons*16, (3,3), activation=None, padding='same')(pool4)
    conv5 = residual_block(conv5, start_neurons*16)
    conv5 = residual_block(conv5, start_neurons*16, True)
    #pool5 = MaxPooling2D((2,2))(conv5)
    #pool5 = Dropout(DropoutRatio)(pool5)

    # 4 -> 4
    #conv6 = Conv2D(start_neurons*32, (3,3), activation=None, padding='same')(pool4)
    #conv6 = residual_block(conv5, start_neurons*32)
    #conv6 = residual_block(conv5, start_neurons*32, True)

    fc1 = GlobalAveragePooling2D(name='avg_pool')(conv5)
    output_layer = Dense(NCATS, activation='softmax', name='output')(fc1)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model


def mobilenet(size):
    model = app.mobilenet.MobileNet(input_shape=(size, size, 1), alpha=1., weights=None, classes=NCATS)
    return model


# def convolution1d_block(x, filters, size, strides=(1,), padding='same', activation=True):
#     x = Conv1D(filters, size, strides=strides, padding=padding)(x)
#     if activation==True: x = BatchActivate(x)
#     return x


# def residual1d_block(blockInput, num_filters=16, batch_activate=False, add_idx=None, prefix=None):
#     x = BatchActivate(blockInput)
#     x = convolution1d_block(x, num_filters, (5,))
#     x = convolution1d_block(x, num_filters, (5,), activation=False)
#     if add_idx:
#         x = Add(name="%s_add_%s"%(prefix, add_idx))([x, blockInput])
#     else:
#         x = Add()([x, blockInput])
#     if batch_activate: x = BatchActivate(x)
#     return x


def rnn(stroke_count=25, conv_neurons=48, rnn_neurons=512, DropoutRatio=0.25):
    input_layer = Input((stroke_count, 3))
    bn1 = BatchNormalization()(input_layer)
    
    conv1 = Conv1D(conv_neurons, (5,), activation='relu', padding='same')(bn1)
    conv1 = BatchNormalization()(conv1)
    conv2 = Conv1D(conv_neurons*2, (5,), activation='relu', padding='same')(conv1)
    conv2 = BatchNormalization()(conv2)
    conv3 = Conv1D(conv_neurons*3, (5,), activation='relu', padding='same')(conv2)
    conv3 = BatchNormalization()(conv3)
    
#     The residual network seems hard to train in this case
#     conv1 = Conv1D(conv_neurons, (5,), activation=None, padding='same')(bn1)
#     conv1 = residual1d_block(conv1, conv_neurons)
#     conv1 = residual1d_block(conv1, conv_neurons, True)

    rnn1 = Bidirectional(LSTM(rnn_neurons, return_sequences = True, recurrent_dropout=DropoutRatio))(conv3)
    rnn1 = Dropout(DropoutRatio)(rnn1)
    rnn2 = Bidirectional(LSTM(rnn_neurons, return_sequences=False, recurrent_dropout=DropoutRatio))(rnn1)
    rnn2 = Dropout(DropoutRatio)(rnn2)
    
    fc1 = Dense(rnn_neurons*2)(rnn2)
    output_layer = Dense(NCATS, activation='softmax')(fc1)
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model
