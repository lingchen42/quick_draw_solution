from helper import *
from models import *
from data_gen import *
from snapshot import *
import keras
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

np.random.seed(seed=1)
tf.set_random_seed(seed=1)

GPU_COUNT = 4
NCATS = 340
NCSVS = 100
STEPS = 5000
EPOCHS = 100
LR = 0.001
DP_DIR = "../data/shuffled_csvs_2_60k/"
OUTMODELPREFIX = "resnet_img_256"
OUTHIST = "weights/resnet_img_256.csv"
size = 256
batchsize = 2048

# build model
print("Bulding model ... ")
model = resnet_img256(size, start_neurons=32, DropoutRatio=0.25)
model = keras.utils.multi_gpu_model(model, gpus=GPU_COUNT, cpu_merge=True, cpu_relocation=False)
#model.load_weights("weights/resnet_big3-Best.h5")
model.compile(optimizer=Adam(lr=LR), loss='categorical_crossentropy',
              metrics=[categorical_crossentropy, categorical_accuracy, top_3_accuracy])
print(model.summary())

# valid data
print("Read validation data...")
valid_df = pd.read_csv(os.path.join("../data/shuffled_csvs/", 'train_k{}.csv.gz'.format(NCSVS - 1)), nrows=34000)
x_valid = df_to_image_array_xd(valid_df, size)
y_valid = keras.utils.to_categorical(valid_df.y, num_classes=NCATS)
print(x_valid.shape, y_valid.shape)
print('Validation array memory {:.2f} GB'.format(x_valid.nbytes / 1024.**3 ))

# train data
print("Create train data generator...")
train_datagen = image_generator_xd(size=size, batchsize=batchsize, ks=range(NCSVS - 1), dp_dir=DP_DIR)

# callbacks
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
