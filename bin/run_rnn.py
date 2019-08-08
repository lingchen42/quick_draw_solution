from helper import *
from models import *
from data_gen import *
from snapshot import *
import keras
from keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

np.random.seed(seed=1)
tf.set_random_seed(seed=1)

GPU_COUNT = 2
NCATS = 340
NCSVS = 100
STEPS = 1000
EPOCHS = 200
LR = 0.0005
STROKE_COUNT = 25
DP_DIR = "../data/shuffled_csvs/"
OUTMODELPREFIX = "gru2"
OUTHIST = "weights/gru2_hist.csv"
batchsize = 256
nsnapshots = 2

# build model 
print("Bulding model ... ")
model = rnn(stroke_count=STROKE_COUNT, conv_neurons=64, rnn_neurons=512, DropoutRatio=0.25)
print(model.summary())
if GPU_COUNT > 1:
    print("Using %s gpus"%GPU_COUNT)
    model = keras.utils.multi_gpu_model(model, gpus=GPU_COUNT, cpu_merge=True, cpu_relocation=False)
model.load_weights("weights/gru2-Best.h5")

#opt = optimizers.SGD(lr=LR, clipnorm=1.)
opt = Adam(lr=LR)
model.compile(optimizer=opt, loss='categorical_crossentropy',
              metrics=[categorical_crossentropy, categorical_accuracy, top_3_accuracy])

# valid data
print("Read validation data...")
valid_df = pd.read_csv(os.path.join(DP_DIR, 'train_k{}.csv.gz'.format(NCSVS - 1)), nrows=34000)
x_valid = df_to_image_array_stroke(valid_df, STROKE_COUNT)
y_valid = keras.utils.to_categorical(valid_df.y, num_classes=NCATS)
print(x_valid.shape, y_valid.shape)
print('Validation array memory {:.2f} GB'.format(x_valid.nbytes / 1024.**3 ))

# train data
print("Create train data generator...")
#STEPS = 200
#train_datagen = image_generator_stroke(batchsize=batchsize, ks=range(5), stroke_count=STROKE_COUNT)
train_datagen = image_generator_stroke(batchsize=batchsize, ks=range(NCSVS - 1), stroke_count=STROKE_COUNT)

# callbacks
print("Build call backs...")
checkpoint = ModelCheckpoint("weights/%s-Best.h5"%OUTMODELPREFIX, monitor='val_loss', verbose=1, 
                             save_best_only=True, mode='min', save_weights_only = True)
reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=10, 
                                   verbose=1, mode='auto', epsilon=0.0001, cooldown=5, min_lr=0.00001)
early = EarlyStopping(monitor="val_loss", 
                      mode="min", 
                      patience=30) # probably needs to be more patient, but kaggle time is limited
callbacks = [checkpoint, early, reduceLROnPlat]

# train
print("Start training...")
hist = model.fit_generator(
    train_datagen, steps_per_epoch=STEPS, epochs=EPOCHS, verbose=1,
    validation_data=(x_valid, y_valid),
    callbacks = callbacks
)

# Saving history
print("Saving history...")
hist_df = pd.DataFrame(hist.history)
print(hist_df)
hist_df.to_csv(OUTHIST, mode="a", header=False, index=False)

# Valid map3
print("Predicting valid data...") 
valid_predictions = model.predict(x_valid, batch_size=128, verbose=1)
map3 = mapk(valid_df[['y']].values, preds2catids(valid_predictions).values)
print('Map3: {:.3f}'.format(map3))