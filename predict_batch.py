import sys
import argparse
import numpy as np
from PIL import Image
import requests
from io import BytesIO
from os.path import join

import keras
from keras.preprocessing import image
import keras.applications as app
from keras.models import Model, load_model
from keras import metrics

target_size = (229, 229)

Class_Index = {
    0:'0',
    1:'1',
    2:'10',
    3:'11',
    4:'12',
    5:'13',
    6:'14',
    7:'15',
    8:'16',
    9:'17',
    10:'18',
    11:'19',
    12:'2',
    13:'3',
    14:'4',
    15:'5',
    16:'6',
    17:'7',
    18:'8',
    19:'9'
}

def decode_predictions(preds, top=3):
    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        result = [(Class_Index[i],) + (pred[i],) for i in top_indices]
        # print(top_indices)
        # print(pred)
        # print(result)
        result.sort(key=lambda x: x[1], reverse=True)
        results.append(result)
    return results

base_model_preprocess = {
    "Xception":app.xception.preprocess_input,
    "InceptionV3":app.inception_v3.preprocess_input,
    "DenseNet201":app.densenet.preprocess_input,
    "Resnet50":app.resnet50.preprocess_input,
    "InceptionResNetV2":app.inception_resnet_v2.preprocess_input
}
    
base_model_size = {
    "Xception":(224, 224),
    "InceptionV3":(299, 299),
    "DenseNet201":(224, 224),
    "Resnet50":(224, 224),
    "InceptionResNetV2":(299,299),
    "NASNetLarge":(224,224)
}


def predict(model,listfilepath,imagepath):
    img = []
    imgfiles = []
    with open(listfilepath,'r') as filelist:
        for i,line in enumerate(filelist):
            if i != 0:
                imgfiles.append(line)
                imgfile = line[:-1] + '.jpg'
                img_single = image.load_img(imagepath+'/'+imgfile,target_size=target_size)     
                x = image.img_to_array(img_single)  
                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)  
                img.append(x)    
    X = np.concatenate([x for x in img])
    y = model.predict(X)

    return y,imgfiles

    
        
if __name__=="__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--imagepath", default="./data",help="path to image")
    a.add_argument("--listfile",default="./list.csv")
    a.add_argument("--basename")
    a.add_argument("--model")
    args = a.parse_args()

    if args.imagepath is None or args.listfile is None or args.basename is None:
        a.print_help()
        sys.exit(1)

    preprocess_input = base_model_preprocess[args.basename]
    target_size = base_model_size[args.basename]

    model = load_model(args.model)
    if args.imagepath is not None and args.listfile is not None:
        preds,imgfiles = predict(model, args.listfile, args.imagepath)
        result = decode_predictions(preds)
        with open("result_"+args.basename+'.csv','w') as fout:
            fout.write('FILE_ID,CATEGORY_ID0,CATEGORY_ID1,CATEGORY_ID2\n')
            for i,filename in enumerate(imgfiles):
                fout.write(filename[:-1])
                for ccls,score in result[i]:
                    fout.write(','+ccls)
                fout.write('\n')
