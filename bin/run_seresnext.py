from helper import *
from models import *
from data_gen import *
from snapshot import *
from accu_adam import AdamAccumulate, NadamAccum, Adam_accumulate, SGDAccum
import keras
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from se_resnext import SEResNext

np.random.seed(seed=1)
tf.set_random_seed(seed=1)

GPU_COUNT = 4
NCATS = 340
NCSVS = 400
STEPS = 800
EPOCHS = 800
LR = 0.01
DP_DIR = "../data/shuffled_csv_all/"
OUTMODELPREFIX = "seresnext"
OUTHIST = "weights/seresnext_hist.csv"
size = 128
batchsize = 64
nsnapshots = 1
NACCUM = 2
START_CSV = 0

# build model
print("Bulding model ... ")
model = SEResNext(input_shape=(size, size, 1), depth=56, classes=NCATS)
print(model.summary())
if GPU_COUNT > 1:
    print("Using %s gpus"%GPU_COUNT)
    model = keras.utils.multi_gpu_model(model, gpus=GPU_COUNT, cpu_merge=True, cpu_relocation=False)
#model.load_weights("weights/seresnext-Best.h5")
opt = Adam(lr=LR)
#print("Use accumlated gradients of %s"%NACCUM)
#opt = SGDAccum(lr=LR, momentum=0.9, accum_iters=NACCUM)
model.compile(optimizer=opt, loss='categorical_crossentropy',
              metrics=[categorical_crossentropy, categorical_accuracy, top_3_accuracy])

# valid data
print("Read validation data...")
valid_df = pd.read_csv(os.path.join(DP_DIR, 'train_k{}.csv.gz'.format(NCSVS - 1)), nrows=17000)
x_valid = df_to_image_array_xd(valid_df, size)
y_valid = keras.utils.to_categorical(valid_df.y, num_classes=NCATS)
print(x_valid.shape, y_valid.shape)
print('Validation array memory {:.2f} GB'.format(x_valid.nbytes / 1024.**3 ))

# train data
print("Create train data generator...")
train_datagen = image_generator_xd(size=size, batchsize=batchsize, ks=range(NCSVS - 1), start_csv=START_CSV, dp_dir=DP_DIR)

# callbacks
print("Build call backs...")
#early_stopping = EarlyStopping(monitor='val_categorical_accuracy',
#                               patience=100, verbose=1)
#snapshotcallback = SnapshotCallbackBuilder(nb_epochs=EPOCHS, nb_snapshots=nsnapshots, init_lr=LR)
#model_checkpoint, lr_schedular, snapshotmodel_checkpoint = \
#    snapshotcallback.get_callbacks(model_prefix=OUTMODELPREFIX)
#callbacks = [early_stopping, model_checkpoint, lr_schedular, snapshotmodel_checkpoint]
checkpoint = ModelCheckpoint("weights/%s-Best.h5"%OUTMODELPREFIX, monitor='val_loss', verbose=1,
                             save_best_only=True, mode='min', save_weights_only = True)
early = EarlyStopping(monitor="val_loss",
                      mode="min",
                      patience=100)
callbacks = [checkpoint, early]

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
