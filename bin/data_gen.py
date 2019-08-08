import os
import ast
import datetime as dt
import cv2
import pandas as pd
import numpy as np
import keras
from keras.preprocessing.sequence import pad_sequences

BASE_SIZE = 256
NCATS = 340

def draw_cv2(raw_strokes, size=256, lw=6, time_color=True):
    img = np.zeros((BASE_SIZE, BASE_SIZE), np.uint8)
    for t, stroke in enumerate(raw_strokes):
        for i in range(len(stroke[0]) - 1):
            color = 255 - min(t, 10) * 13 if time_color else 255
            _ = cv2.line(img, (stroke[0][i], stroke[1][i]),
                         (stroke[0][i + 1], stroke[1][i + 1]), color, lw)
    if size != BASE_SIZE:
        return cv2.resize(img, (size, size))
    else:
        return img


def image_generator_xd(size, batchsize, ks, start_csv=0,
                       lw=6, time_color=True,
                       dp_dir="../data/shuffled_csvs/"):
    while True:
        #for k in np.random.permutation(ks):
        for k in ks:
            if k >= start_csv:
                print(k)
                filename = os.path.join(dp_dir, 'train_k{}.csv.gz'.format(k))
                for df in pd.read_csv(filename, chunksize=batchsize):
                    df['drawing'] = df['drawing'].apply(ast.literal_eval)
                    x = np.zeros((len(df), size, size, 1))
                    for i, raw_strokes in enumerate(df.drawing.values):
                        x[i, :, :, 0] = draw_cv2(raw_strokes, size=size, lw=lw,
                                             time_color=time_color)
                    #x = preprocess_input(x).astype(np.float32)
                    x = x.astype(np.float32)
                    x = x / 255.
                    y = keras.utils.to_categorical(df.y, num_classes=NCATS)
                    yield x, y


def df_to_image_array_xd(df, size, lw=6, time_color=True):
    df['drawing'] = df['drawing'].apply(ast.literal_eval)
    x = np.zeros((len(df), size, size, 1))
    for i, raw_strokes in enumerate(df.drawing.values):
        x[i, :, :, 0] = draw_cv2(raw_strokes, size=size, lw=lw, time_color=time_color)
#    x = preprocess_input(x).astype(np.float32)
    x = x.astype(np.float32)
    x = x / 255.
    return x


def _stack_it(raw_strokes, stroke_count):
    """preprocess the string and make 
    a standard Nx3 stroke vector"""
    #stroke_vec = literal_eval(raw_strokes) # string->list
    stroke_vec = raw_strokes
    # unwrap the list
    in_strokes = [(xi,yi,i)  
     for i,(x,y) in enumerate(stroke_vec) 
     for xi,yi in zip(x,y)]
    c_strokes = np.stack(in_strokes)
    # replace stroke id with 1 for continue, 2 for new
    c_strokes[:,2] = [1]+np.diff(c_strokes[:,2]).tolist()
    c_strokes[:,2] += 1 # since 0 is no stroke
    # pad the strokes with zeros
    return pad_sequences(c_strokes.swapaxes(0, 1), 
                         maxlen=stroke_count, 
                         padding='post').swapaxes(0, 1)



def image_generator_stroke(batchsize, ks, stroke_count, start_csv=0,
                           dp_dir="../data/shuffled_csvs/"):

    while True:
        #for k in np.random.permutation(ks):
        for k in range(ks):
            if k >= start_csv:
                print("train: ", k)
                filename = os.path.join(dp_dir, 'train_k{}.csv.gz'.format(k))
                for df in pd.read_csv(filename, chunksize=batchsize):
                    df['drawing'] = df['drawing'].apply(ast.literal_eval)
                    x = np.zeros((len(df), stroke_count, 3))
                    for i, raw_strokes in enumerate(df.drawing.values):
                        x[i, :, :] = _stack_it(raw_strokes, stroke_count)
                    y = keras.utils.to_categorical(df.y, num_classes=NCATS)
                    yield x, y


def df_to_image_array_stroke(df, stroke_count):
    df['drawing'] = df['drawing'].apply(ast.literal_eval)
    x = np.zeros((len(df), stroke_count, 3))
    for i, raw_strokes in enumerate(df.drawing.values):
        x[i, :, :] = _stack_it(raw_strokes, stroke_count)
    return x


def img_and_stroke_generator(size, batchsize, ks, stroke_count, lw=6, time_color=True, 
                             dp_dir="../data/shuffled_csvs/"):
    while True:
        for k in np.random.permutation(ks):
            filename = os.path.join(dp_dir, 'train_k{}.csv.gz'.format(k))
            for df in pd.read_csv(filename, chunksize=batchsize):
                df['drawing'] = df['drawing'].apply(ast.literal_eval) 
                print(df.index[0])
                x1 = np.zeros((len(df), size, size, 1))
                x2 = np.zeros((len(df), stroke_count, 3))
                for i, raw_strokes in enumerate(df.drawing.values):
                    x1[i, :, :, 0] = draw_cv2(raw_strokes, size=size, lw=lw,
                                             time_color=time_color)
                    x1 = x1.astype(np.float32)
                    x1 = x1 / 255.
                    x2[i, :, :] = _stack_it(raw_strokes, stroke_count)
                    y = keras.utils.to_categorical(df.y, num_classes=NCATS)
                    yield [x1, x2], y
