import os
import sys
import glob
import argparse
import matplotlib.pyplot as plt

import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import __version__
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD,Adam


IM_WIDTH, IM_HEIGHT = 299, 299  # fixed size for InceptionV3
NB_EPOCHS = 3
BAT_SIZE = 32
FC_SIZE = 1024
NB_IV3_LAYERS_TO_FREEZE = 172

#epochs为全局的超参数epoch，epcho表示当前epoch
#Learning rate 不要固定在lr=0.0001，最好是阶梯下降
epochs = 0
def lr_schedule(epoch):
    #Learning Rate Schedule
    lr = 1e-3
    if epoch >= epochs * 0.9:
        lr *= 0.5e-3
    elif epoch >= epochs * 0.8:
        lr *= 1e-3
    elif epoch >= epochs * 0.6:
        lr *= 1e-2
    elif epoch >= epochs * 0.4:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

def get_nb_files(directory):
    """Get number of files by searching directory recursively"""
    if not os.path.exists(directory):
        return 0
    cnt = 0
    for r, dirs, files in os.walk(directory):
        for dr in dirs:
            cnt += len(glob.glob(os.path.join(r, dr + "/*")))
    return cnt


def setup_to_transfer_learn(model, base_model):
    """Freeze all layers and compile the model"""
    for layer in base_model.layers:
        layer.trainable = False
    #建议使用Adam优化器
    model.compile(optimizer='adam',
                    loss='categorical_crossentropy', metrics=['accuracy'])


def add_new_last_layer(base_model, nb_classes):
    """Add last layer to the convnet
    Args:
        base_model: keras model excluding top
        nb_classes: # of classes
    Returns:
        new keras model with last layer
    """
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(FC_SIZE, activation='relu')(x)  # new FC layer, random init
    predictions = Dense(nb_classes, activation='softmax')(x)  # new softmax layer
    model = Model(outputs=predictions,inputs=base_model.input)
    return model


#Fine Tune建议可以逐步打开层效果更好。例如40%epoch开前30%的层，而不是一次性全部打开。
def setup_to_finetune(model):
    """Freeze the bottom NB_IV3_LAYERS and retrain the remaining top layers.
    note: NB_IV3_LAYERS corresponds to the top 2 inception blocks in the inceptionv3 arch
    Args:
        model: keras model
    """
    #这行没用
    # for layer in model.layers[:NB_IV3_LAYERS_TO_FREEZE]:
    #     layer.trainable = False
    for layer in model.layers[NB_IV3_LAYERS_TO_FREEZE:]:
        layer.trainable = True
    #用Adam试试
    model.compile(optimizer=Adam(lr=lr_schedule(0)),
                    loss='categorical_crossentropy', metrics=['accuracy'])


def train(args):
    """Use transfer learning and fine-tuning to train a network on a new dataset"""
    nb_train_samples = get_nb_files(args.train_dir)
    nb_classes = len(glob.glob(args.train_dir + "/*"))
    nb_val_samples = get_nb_files(args.val_dir)
    nb_epoch = int(args.nb_epoch)

    #Set global epochs
    epochs = nb_epoch

    batch_size = int(args.batch_size)

    # data prep
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        featurewise_center=True,
        samplewise_center=True,
    )

    # Test不需要数据增强
    test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        featurewise_center=True,
        samplewise_center=True,
    )

    train_generator = train_datagen.flow_from_directory(
        args.train_dir,
        target_size=(IM_WIDTH, IM_HEIGHT),
        batch_size=batch_size,
    )

    validation_generator = test_datagen.flow_from_directory(
        args.val_dir,
        target_size=(IM_WIDTH, IM_HEIGHT),
        batch_size=batch_size,
    )

    check = ModelCheckpoint('./history/modeltmp.model',
                monitor='val_loss',
                verbose=0,
                save_best_only=False,
                save_weights_only=False,
                mode='auto',
                period=1)

    #stop可以取消试试，因为过早的结束可能不是最优解。patience为0是不行的。
    stop = EarlyStopping(monitor='val_loss',
                min_delta=0,
                patience=10,
                verbose=0,
                mode='auto')


    # setup model
    # include_top=False excludes final FC layer
    base_model = InceptionV3(weights='imagenet', include_top=False)

    #Inception V4/Xcception/ DenseNet 效果更好
    base_model = keras.applications.DenseNet121(weights='imagenet', include_top=False)
    # base_model = keras.applications.Xception(weights='imagenet', include_top=False)
    # base_model = keras.applications.InceptionResNetV2(weights='imagenet', include_top=False)

    model = add_new_last_layer(base_model, nb_classes)

    # transfer learning
    setup_to_transfer_learn(model, base_model)

    history_tl = model.fit_generator(
        train_generator,
        epochs=nb_epoch,
        steps_per_epoch = nb_train_samples/batch_size,
        verbose = 1,
        validation_data=validation_generator,
        class_weight='auto',
        shuffle = True,
        callbacks = [check, stop]
        )

    model.save("transfered"+args.output_model_file)

    # 参照下面代码加一下TensorBoard
    # from keras.callbacks import TensorBoard
    # callbacks = [checkpoint, lr_reducer, lr_scheduler,
    #              TensorBoard(log_dir='./TB_logdir/ResNet_GaussianNoise', write_images=False)]


    # fine-tuning
    setup_to_finetune(model)
    print("start fine tune")
    history_ft = model.fit_generator(
        train_generator,
        verbose = 1,
        epochs=nb_epoch,
        steps_per_epoch = nb_train_samples/batch_size,
        validation_data=validation_generator,
        class_weight='auto',
        shuffle = True,
        callbacks = [check, stop]
        )

    model.save(args.output_model_file)

    if args.plot:
        plot_training(history_ft)


def plot_training(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))

    plt.plot(epochs, acc, 'r.')
    plt.plot(epochs, val_acc, 'r')
    plt.title('Training and validation accuracy')

    plt.figure()
    plt.plot(epochs, loss, 'r.')
    plt.plot(epochs, val_loss, 'r-')
    plt.title('Training and validation loss')
    plt.show()


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--train_dir")
    a.add_argument("--val_dir")
    a.add_argument("--nb_epoch", default=NB_EPOCHS)
    a.add_argument("--batch_size", default=BAT_SIZE)
    a.add_argument("--output_model_file", default="inceptionv3-ft.model")
    a.add_argument("--plot", action="store_true")

    args = a.parse_args()
    if args.train_dir is None or args.val_dir is None:
        a.print_help()
        sys.exit(1)

    if (not os.path.exists(args.train_dir)) or (not os.path.exists(args.val_dir)):
        print("directories do not exist")
        sys.exit(1)
   train(args)
