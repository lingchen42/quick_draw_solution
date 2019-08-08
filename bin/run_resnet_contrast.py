# %matplotlib inline
# import matplotlib.pyplot as plt
import os
import cv2
import ast
import sys
sys.path.insert(0, "../bin/")
import importlib
import pandas as pd
import numpy as np
from keras.applications.resnet50 import ResNet50
from keras.callbacks import EarlyStopping, ModelCheckpoint

from helper import *
from models import resnet_img256
import keras
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Dense, GlobalAveragePooling2D
from keras.callbacks import EarlyStopping
from accu_adam import AdamAccumulate, NadamAccum, Adam_accumulate, SGDAccum
colors = ['#222222', '#F3C300', '#875692', '#F38400', '#A1CAF1', '#BE0032', '#C2B280', '#848482', 
          '#008856', '#E68FAC', '#0067A5', '#F99379', '#604E97', '#F6A600', '#B3446C', '#DCD300', '#882D17', 
          '#8DB600', '#654522', '#E25822', '#2B3D26']
rgb_colors = [tuple(int(c[1:][i:i+2], 16) for i in (0, 2 ,4)) for c in colors]


BASE_SIZE = 256
SIZE = 128
GPU_COUNT = 2
LR = 0.0005
NCATS = 340
NCSVS = 400
BATCHSIZE = 200
DP_DIR= "../data/shuffled_csv_all/"
OUTMODELPREFIX = "keras_resnet50_img128_contrast2"
STEPS = 1000
EPOCHS = 800
START_TRAIN = 92
NACCUM = 4


def draw_cv2(raw_strokes, size=128, lw=3, time_color=True):
    img = np.zeros((BASE_SIZE, BASE_SIZE, 3), np.uint8) + 255
    for t, stroke in enumerate(raw_strokes):
        for i in range(len(stroke[0]) - 1):
           # color = min(t, 20) * 10 if time_color else 0
            c = int(t/4) % len(rgb_colors)  # every 4 strokes, change a color
            color = rgb_colors[c]
            _ = cv2.line(img, (stroke[0][i], stroke[1][i]),
                         (stroke[0][i + 1], stroke[1][i + 1]), color, lw)
    if size != BASE_SIZE:
        return cv2.resize(img, (size, size))
    else:
        return img


def image_generator_xd(size, batchsize, ks,
                       lw=3, time_color=True,
                       dp_dir="../data/shuffled_csvs_2_60k/"):
    while True:
        #for k in np.random.permutation(ks):
        for k in range(START_TRAIN, ks):
            print("train data: No.", k)
            filename = os.path.join(dp_dir, 'train_k{}.csv.gz'.format(k))

            for df in pd.read_csv(filename, chunksize=batchsize):
                df['drawing'] = df['drawing'].apply(ast.literal_eval)
                x = np.zeros((len(df), size, size, 3))
                for i, raw_strokes in enumerate(df.drawing.values):
                    x[i, :, :, :] = draw_cv2(raw_strokes, size=size, lw=lw,
                                             time_color=time_color)
                x = x.astype(np.float32)
                x = x / 255.
                y = keras.utils.to_categorical(df.y, num_classes=NCATS)
                yield x, y


def df_to_image_array_xd(df, size, lw=3, time_color=True):
    df['drawing'] = df['drawing'].apply(ast.literal_eval)
    x = np.zeros((len(df), size, size, 3))
    for i, raw_strokes in enumerate(df.drawing.values):
        x[i, :, :, :] = draw_cv2(raw_strokes, size=size, lw=lw, time_color=time_color)
    x = x.astype(np.float32)
    x = x / 255.
    return x


# build model
print("Bulding model ... ")
model = ResNet50(include_top=False, input_shape=(SIZE, SIZE, 3))
x = model.outputs[0]
x = GlobalAveragePooling2D()(x)
x = Dense(NCATS, activation='softmax')(x)
model = Model(inputs=model.inputs, outputs=x)
print(model.summary())

model = keras.utils.multi_gpu_model(model, gpus=GPU_COUNT, cpu_merge=True, cpu_relocation=False)
model.load_weights("weights/keras_resnet50_img128_contrast-Best.h5")
opt = SGDAccum(lr=LR, momentum=0.9, accum_iters=NACCUM)
model.compile(optimizer=opt, loss='categorical_crossentropy',
              metrics=[categorical_crossentropy, categorical_accuracy, top_3_accuracy])

# valid data
print("Read validation data...")
valid_df = pd.read_csv(os.path.join("../data/shuffled_csv_all/", 'train_k{}.csv.gz'.format(NCSVS - 1)), nrows=17000)
x_valid = df_to_image_array_xd(valid_df, SIZE)
y_valid = keras.utils.to_categorical(valid_df.y, num_classes=NCATS)

print("Create train data generator...")
train_datagen = image_generator_xd(size=SIZE, batchsize=BATCHSIZE, ks=(NCSVS - 1),
                                   dp_dir=DP_DIR)

print("Build call backs...")
checkpoint = ModelCheckpoint("weights/%s-Best.h5"%OUTMODELPREFIX, monitor='val_loss', verbose=1,
                             save_best_only=True, mode='min', save_weights_only = True)
# reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=10,
#                                    verbose=1, mode='auto', epsilon=0.0001, cooldown=5, min_lr=0.00001)
early = EarlyStopping(monitor="val_loss",
                      mode="min",
                      patience=40)
callbacks = [checkpoint, early]

# train
print("Start training...")
hist = model.fit_generator(
    train_datagen, steps_per_epoch=STEPS, epochs=EPOCHS, verbose=1,
    validation_data=(x_valid, y_valid),
    callbacks = callbacks
)
