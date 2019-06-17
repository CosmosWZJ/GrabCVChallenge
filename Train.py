from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Flatten, Dense, Dropout
from tensorflow.python.keras.applications.resnet50 import ResNet50
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import ReadData
DATASET_PATH  = "C:\\Users\\test\\PycharmProjects\\grabchallenge"
# image size
IMAGE_SIZE = (256, 256)
# categories
NUM_CLASSES = 196

BATCH_SIZE = 16
# number of freeze layer
FREEZE_LAYERS = 2
# Epoch number
NUM_EPOCHS = 5
# output name
WEIGHTS_FINAL = 'model-resnet50-final1.h5'
# use data augmentation to get more images
train_datagen = ImageDataGenerator(rotation_range=20,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   channel_shift_range=10,
                                   horizontal_flip=True,
                                   fill_mode='nearest')
# train and validate 4:1, since the original training set is distributed evenly
train_frame = ReadData.read_train_data()
test_frame = train_frame.iloc[int(8144*0.8):]
train_frame = train_frame.iloc[:int(8144*0.8)]

train_batches = train_datagen.flow_from_dataframe(train_frame, DATASET_PATH + '/cars_train',x_col='fileName', y_col='class',
                                                  target_size=IMAGE_SIZE,
                                                  interpolation='bicubic',
                                                  class_mode='categorical',
                                                  shuffle=True,
                                                  batch_size=BATCH_SIZE)
valid_datagen = ImageDataGenerator()
valid_batches = valid_datagen.flow_from_dataframe(test_frame, DATASET_PATH + '/cars_train',x_col='fileName', y_col='class',
                                                  interpolation='bicubic',
                                                  class_mode='categorical',
                                                  shuffle=False,
                                                  batch_size=BATCH_SIZE)

net = ResNet50(include_top=False, weights='imagenet', input_tensor=None,
               input_shape=(IMAGE_SIZE[0],IMAGE_SIZE[1],3))
x = net.output
x = Flatten()(x)

# add DropOut layer
x = Dropout(0.5)(x)

# add Dense layer，use softmax function to generate probability of different categories
output_layer = Dense(NUM_CLASSES, activation='softmax', name='softmax')(x)

# set freeze layer
net_final = Model(inputs=net.input, outputs=output_layer)
for layer in net_final.layers[:FREEZE_LAYERS]:
    layer.trainable = False
for layer in net_final.layers[FREEZE_LAYERS:]:
    layer.trainable = True

# use Adam optimizer，fine-tuning with low learning rate
net_final.compile(optimizer=Adam(lr=1e-5),
                  loss='categorical_crossentropy', metrics=['accuracy'])

# print network structure
#print(net_final.summary())

# train model
net_final.fit_generator(train_batches,
                        steps_per_epoch = train_batches.samples // BATCH_SIZE,
                        validation_data = valid_batches,
                        validation_steps = valid_batches.samples // BATCH_SIZE,
                        epochs = NUM_EPOCHS)

# store model
net_final.save(WEIGHTS_FINAL)