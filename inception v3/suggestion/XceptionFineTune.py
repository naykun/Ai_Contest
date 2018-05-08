from __future__ import print_function
import time
script_start_time = time.time()
import keras
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os

from keras.callbacks import TensorBoard
from keras.optimizers import Adam
import Utilities.utility_img as img_reader
import Utilities.ExportResult as export_res
from Utilities.AnalizeResult import predict_classes
from sklearn.metrics import confusion_matrix, classification_report

# Training parameters
batch_size = 32  # orig paper trained all networks with batch_size=128
epochs = 30
data_augmentation = False
split_ratio = 0.8 # Train = [0:split_ratio], Val = [split_ratio:]
num_classes = -1
log_dir = './ana/OYZH/p13_decay0.001_moreDenselayer_layer'
image_path = '/home/amax/data2/GVS2SA2018_Processed/224_Vidi_Train/'
pre_train_moduel_path = 'saved_models/pretrai_13class_Xception.h5'
# Subtracting pixel mean improves accuracy
subtract_pixel_mean = True
max_img_per_folder = 99999

#Read Data
def shuffle_data(data, label, path):
    num_example = data.shape[0]
    arr = np.arange(num_example)
    np.random.shuffle(arr)
    data = data[arr]
    label = label[arr]
    path = path[arr]
    return data, label, path

# Shuffle + Split -> train + val
x_, y_, pathes, num_classes = img_reader.read_img(image_path, max_img_per_folder)
x_, y_, pathes = shuffle_data(x_, y_, pathes)
split_idx = int(len(y_) * split_ratio)
x_train, y_train, pathes_train = x_[0:split_idx], y_[0:split_idx], pathes[0:split_idx]
x_val, y_val,pathes_val = x_[split_idx:], y_[split_idx:],pathes[split_idx:]
IMAGES_TRAIN = img_reader.img_class(x_train, y_train, pathes_train)
IMAGES_VAL = img_reader.img_class(x_val, y_val, pathes_val)


# Input image dimensions.
# input_shape = IMAGES_TRAIN.data.shape[1:]

# Normalize data.
IMAGES_TRAIN.data = IMAGES_TRAIN.data.astype('float32') / 255
IMAGES_VAL.data = IMAGES_VAL.data.astype('float32') / 255

# If subtract pixel mean is enabled
if subtract_pixel_mean:
    x_train_mean = np.mean(IMAGES_TRAIN.data, axis=0)
    IMAGES_TRAIN.data -= x_train_mean
    IMAGES_VAL.data -= x_train_mean

# Convert class vectors to binary class matrices.
IMAGES_TRAIN.label = keras.utils.to_categorical(IMAGES_TRAIN.label, num_classes)
IMAGES_VAL.label = keras.utils.to_categorical(IMAGES_VAL.label, num_classes)

def lr_schedule(epoch):
    #Learning Rate Schedule
    lr = 1e-3
    if epoch >= epochs * 0.9:
        lr *= 0.5e-3
    elif epoch >= epochs * 0.8:
        lr *= 1e-3
    elif epoch >= epochs * 0.6:
        freeze_layer(model, len(model.layers))
        lr *= 1e-2
    elif epoch >= epochs * 0.4:
        freeze_layer(model, len(model.layers) / 2)
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

# model = DN121.DenseNet(reduction=0.5, classes=num_classes)
model = keras.applications.xception.Xception(include_top=True,
                                            weights=None,
                                            input_tensor=None,
                                            input_shape=None,
                                            pooling=None,
                                            classes=14)

def freeze_layer(model, jump_layer):
    layer_num = len(model.layers)
    for layer in model.layers[:(layer_num - jump_layer)]:
        layer.trainable = False
    for layer in model.layers[(layer_num - jump_layer):]:
        layer.trainable = True

old_model =  keras.models.load_model(pre_train_moduel_path)
model = keras.models.Model(inputs=old_model.input, outputs=old_model.get_layer('block14_sepconv2_act').output)

# layer_num = len(model.layers)
#
# for idx in range(layer_num):
#     model.layers[idx].set_weights(old_model.layers[idx].get_weights())
#
# for layer in model.layers[:(layer_num - 9)]:
#    layer.trainable = False
# for layer in model.layers[(layer_num - 9):]:
#    layer.trainable = True

# add a global spatial average pooling layer
x = model.output
x = keras.layers.GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = keras.layers.Dense(1024, activation='relu')(x)
x = keras.layers.Dense(512, activation='relu')(x)
# and a logistic layer -- let's say we have num_classes classes
predictions = keras.layers.Dense(num_classes, activation='softmax')(x)

model = keras.models.Model(inputs=model.input, outputs=predictions)
freeze_layer(model, 1)

# sgd = keras.optimizers.SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
# model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=lr_schedule(0)),
              metrics=['accuracy'])
model.summary()

print(IMAGES_TRAIN.data.shape[0], 'train samples')
print(IMAGES_VAL.data.shape[0], 'validate samples')
print('layer num: %d', len(model.layers) )
print('x_train shape:', IMAGES_TRAIN.data.shape)
print('y_train shape:', IMAGES_TRAIN.label.shape)

# Prepare model model saving directory.
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'pretrain_13_to_6.h5'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

# Prepare callbacks for model saving and for learning rate adjustment.
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=True)

lr_scheduler = LearningRateScheduler(lr_schedule)

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)

callbacks = [checkpoint, lr_reducer, lr_scheduler, TensorBoard(log_dir=log_dir,write_images=False)]

# Run training, with or without data augmentation.
if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(IMAGES_TRAIN.data, IMAGES_TRAIN.label,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(IMAGES_VAL.data, IMAGES_VAL.label),
              shuffle=True,
              callbacks=callbacks)
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        # set input mean to 0 over the dataset
        featurewise_center=True,
        # set each sample mean to 0
        samplewise_center=True,
        # divide inputs by std of dataset
        featurewise_std_normalization=True,
        # divide each input by its std
        samplewise_std_normalization=True,
        # apply ZCA whitening
        zca_whitening=False,
        # randomly rotate images in the range (deg 0 to 180)
        rotation_range=0,
        # randomly shift images horizontally
        width_shift_range=0.1,
        # randomly shift images vertically
        height_shift_range=0.1,
        # randomly flip images
        horizontal_flip=True,
        # randomly flip images
        vertical_flip=False)

    # Compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(IMAGES_TRAIN.data)

    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(datagen.flow(IMAGES_TRAIN.data, IMAGES_TRAIN.label, batch_size=batch_size),
                        validation_data=(IMAGES_VAL.data, IMAGES_VAL.label),
                        epochs=epochs, verbose=1, workers=4,
                        callbacks=callbacks)

# Predict
pre_ans_prob = model.predict(IMAGES_VAL.data, batch_size=32)
pre_ans_label = predict_classes(pre_ans_prob)

# Save outputs
# cm = confusion_matrix(y_val, pre_ans_label)
# np.savetxt('./Outputs/DenseNet_'+str(epochs) + 'Epochs_' + str(batch_size) +'Batches' + '_Train_ConfusionMatrix.csv', cm, delimiter=',')
# print(cm)
# print(classification_report(y_val,np.array(pre_ans_label)))
# print('Total time cost: %f ' % (time.time() - script_start_time) )
# export_res.export_xml_result('./Outputs/DenseNet_'+str(epochs) + 'Epochs_' + str(batch_size) +'Batches' + '_Train_Results.xls',
# [IMAGES_VAL.path, np.argmax(IMAGES_VAL.label,axis=1),pre_ans_prob,pre_ans_prob[0],pre_ans_prob[1],pre_ans_prob[2],pre_ans_prob[3],pre_ans_prob[4],pre_ans_prob[5]                                          ],
# ['File_Path','Right_Label','Predict','L0','L1','L2','L3','L4','L5']
# )
