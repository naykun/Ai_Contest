import os
import sys
import glob
import argparse

from keras import __version__
from keras import metrics
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.models import load_model

IM_WIDTH, IM_HEIGHT = 299, 299  # fixed size for InceptionV3
NB_EPOCHS = 3
BAT_SIZE = 32
FC_SIZE = 1024
NB_IV3_LAYERS_TO_FREEZE = 172


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


def evaluate(args):
    """evaluate on validate dataset"""
    nb_classes = len(glob.glob(args.val_dir + "/*"))
    nb_val_samples = get_nb_files(args.val_dir)

    # data prep
    test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )
    validation_generator = test_datagen.flow_from_directory(
        args.val_dir,
        target_size=(IM_WIDTH, IM_HEIGHT),
        batch_size=BAT_SIZE,
    )

    # load model
    valid_model = load_model(args.model)
    valid_model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),
                    loss='categorical_crossentropy',metrics=['accuracy',top_5_categorical_accuracy])
    results = valid_model.evaluate_generator(validation_generator)
    for name,result in zip(valid_model.metrics_names,results):
        print(name,":",result)

    

if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--val_dir")
    a.add_argument("--model")
    a.add_argument("--plot", action="store_true")

    args = a.parse_args()
    if  args.model is None:
        a.print_help()
        sys.exit(1)

    if (not os.path.exists(args.val_dir)):
        print("directories do not exist")
        sys.exit(1)

    evaluate(args)
