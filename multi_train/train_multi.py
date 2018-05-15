import os
import sys
import glob
import argparse
import matplotlib.pyplot as plt

import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import __version__
import keras.applications as app
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD,Adam
from keras.callbacks import TensorBoard,LearningRateScheduler,ReduceLROnPlateau
from keras import metrics

IM_WIDTH, IM_HEIGHT = 299, 299  # fixed size for InceptionV3
NB_EPOCHS = 3
BAT_SIZE = 32
FC_SIZE = 1024
NB_LAYERS_TO_FREEZE = 172

#epochs为全局的超参数epoch，epcho表示当前epoch
#Learning rate 不要固定在lr=0.0001，最好是阶梯下降
epochs = 0

def get_nb_files(directory):
    """Get number of files by searching directory recursively"""
    if not os.path.exists(directory):
        return 0
    cnt = 0
    for r, dirs, files in os.walk(directory):
        for dr in dirs:
            cnt += len(glob.glob(os.path.join(r, dr + "/*")))
    return cnt

def top_5_categorical_accuracy(ytrue, ypred):
    return metrics.top_k_categorical_accuracy(ytrue, ypred, k=5)

def setup_to_transfer_learn(model, base_model):
    """Freeze all layers and compile the model"""
    for layer in base_model.layers:
        layer.trainable = False
    #建议使用Adam优化器
    model.compile(optimizer='adam',
                    loss='categorical_crossentropy', metrics=['accuracy',metrics.top_k_categorical_accuracy])


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
# 
def setup_to_finetune(model,rate):
    """Freeze the bottom NB_IV3_LAYERS and retrain the remaining top layers.
    note: NB_IV3_LAYERS corresponds to the top 2 inception blocks in the inceptionv3 arch
    Args:
        model: keras model
    """
    #这行没用
    # for layer in model.layers[:NB_IV3_LAYERS_TO_FREEZE]:
    #     layer.trainable = False
    num_of_layers = len(model.layers)
    for layer in model.layers[int(num_of_layers*rate):]:
        layer.trainable = True
    #用Adam试试
    model.compile(optimizer='adam',
                    loss='categorical_crossentropy', metrics=['accuracy',metrics.top_k_categorical_accuracy])


base_model_funcs = {
    "Xception":app.Xception,
    "InceptionV3":app.InceptionV3,
    "DenseNet201":app.DenseNet201,
    "Resnet50":app.ResNet50,
    "InceptionResNetV2":app.InceptionResNetV2,
    "NASNetLarge":app.NASNetLarge
}
base_model_preprocess = {
    "Xception":app.xception.preprocess_input,
    "InceptionV3":app.inception_v3.preprocess_input,
    "DenseNet201":app.densenet.preprocess_input,
    "Resnet50":app.resnet50.preprocess_input,
    "InceptionResNetV2":app.inception_resnet_v2.preprocess_input,
    "NASNetLarge":app.nasnet.preprocess_input
}
    
base_model_size = {
    "Xception":(224, 224),
    "InceptionV3":(299, 299),
    "DenseNet201":(224, 224),
    "Resnet50":(224, 224),
    "InceptionResNetV2":(299,299),
    "NASNetLarge":(331,331)
}

def get_basemodel(basename):
    func = base_model_funcs[basename]
    basemodel = func(weights='imagenet', include_top=False)
    return basemodel


def train(args):
    """Use transfer learning and fine-tuning to train a network on a new dataset"""
    nb_train_samples = get_nb_files(args.train_dir)
    nb_classes = len(glob.glob(args.train_dir + "/*"))
    nb_val_samples = get_nb_files(args.val_dir)
    nb_epoch = int(args.nb_epoch)

    #Set global epochs
    epochs = nb_epoch

    batch_size = int(args.batch_size)
    preprocess_input = base_model_preprocess[args.basename]

    IM_WIDTH, IM_HEIGHT = base_model_size[args.basename]


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

    checkpoint = ModelCheckpoint(
                args.output_model_path+'_modeltmp'+'.model',
                monitor='val_loss',
                verbose=0,
                save_best_only=False,
                save_weights_only=False,
                mode='auto',
                period=1)

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
        
    #stop可以取消试试，因为过早的结束可能不是最优解。patience为0是不行的。
    # stop = EarlyStopping(monitor='val_loss',
    #             min_delta=0,
    #             patience=10,
    #             verbose=0,
    #             mode='auto')

    learnrt = LearningRateScheduler(lr_schedule)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=0.001)

    callbacks = [checkpoint, learnrt,reduce_lr,
                 TensorBoard(log_dir='./TB_logdir/'+args.basename+"_"+str(args.nb_epoch)+"_"+str(args.batch_size)+'/', write_images=False)]


    # setup model
    # include_top=False excludes final FC layer

    base_model = get_basemodel(args.basename)
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
        # shuffle = True,
        callbacks = callbacks
        )

    model.save(args.output_model_path+"_transfered"+'.model')

    # fine-tuning
    print("start fine tune")
    for freeze_iter in range(1,4):
        setup_to_finetune(model,1-0.1*freeze_iter)
        history_ft = model.fit_generator(
            train_generator,
            verbose = 1,
            epochs=nb_epoch,
            steps_per_epoch = nb_train_samples/batch_size,
            validation_data=validation_generator,
            class_weight='auto',
            # shuffle = True,
            callbacks = callbacks
            )

    model.save(args.output_model_path+'.model')

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
    a.add_argument("--basename")
    a.add_argument("--nb_epoch", default=NB_EPOCHS)
    a.add_argument("--batch_size", default=BAT_SIZE)
    a.add_argument("--output_model_path",default = "./history")
    a.add_argument("--plot", action="store_true")

    args = a.parse_args()
    if args.train_dir is None or args.val_dir is None or args.output_model_path is None or args.basename is None:
        a.print_help()
        sys.exit(1)

    if (not os.path.exists(args.train_dir)) or (not os.path.exists(args.val_dir)):
        print("directories do not exist")
        sys.exit(1)

    args.output_model_path+='/'+args.basename+'/'+args.basename+"_"+str(args.nb_epoch)+"_"+str(args.batch_size)

    if (not os.path.exists(args.output_model_path)):
        os.makedirs(args.output_model_path)

    args.output_model_path+='/'+args.basename+"_"+str(args.nb_epoch)+"_"+str(args.batch_size)
    train(args)
