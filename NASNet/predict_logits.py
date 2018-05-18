import sys
import argparse
import numpy as np
from PIL import Image
import requests
from io import BytesIO
from os.path import join

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torchvision import transforms, models
import pretrainedmodels

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

normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
preprocess_input = transforms.Compose([  
            transforms.Resize(350),  
            transforms.CenterCrop(331),  
            transforms.ToTensor(),  
            normalize  
        ])
    


def predict(model,listfilepath,imagepath):
    img = []
    imgfiles = []
    with open(listfilepath,'r') as filelist:
        for i,line in enumerate(filelist):
            if i != 0:
                imgfiles.append(line)
                imgfile = line[:-1] + '.jpg'
                img_single = Image.open(imagepath+'/'+imgfile) 
                img_single = img_single.convert('RGB')    
                x = preprocess_input(sim_single)  
                img.append(x)    
    numel = sum([x.numel() for x in img])
    storage = img[0].storage()._new_shared(numel)
    out = img[0].new(storage)
    inputs = torch.stack(batch, 0, out=out)
    # X = np.concatenate([x for x in img])
    model.eval()
    with torch.no_grad():
        if use_gpu:  
            inputs = Variable(inputs.cuda())
        else:  
            inputs = Variable(inputs)
        # forward  
        y = model(inputs)
        y = y.cpu()  
        y = y.data.numpy()
        
    return y,imgfiles

    
        
if __name__=="__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--imagepath", default="./data",help="path to image")
    a.add_argument("--listfile",default="./list.csv")
    a.add_argument("--model")
    args = a.parse_args()

    if args.imagepath is None or args.listfile is None :
        a.print_help()
        sys.exit(1)

    model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
    dim_feats = model.last_linear.in_features 
    nb_classes = 20
    model.last_linear = nn.Linear(dim_feats, nb_classes)

    if use_gpu:
        model = torch.nn.DataParallel(model).cuda()
    if os.path.isfile(args.model):
        print("=> loading checkpoint '{}'".format(args.model))
        checkpoint = torch.load(args.resume)
        if type(checkpoint)==dict:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        print("=> no checkpoint found at '{}'".format(args.model))
        return
    cudnn.benchmark = True

    if args.imagepath is not None and args.listfile is not None:
        preds,imgfiles = predict(model, args.listfile, args.imagepath)
        result = decode_predictions(preds)
        with open("result_"+args.basename+'.csv','w') as fout:
            fout.write('FILE_ID,CATEGORY_ID0,CATEGORY_ID1,CATEGORY_ID2\n')
            for i,filename in enumerate(imgfiles):
                fout.write(filename[:-1])
                for score in preds[i]:
                    fout.write(','+str(score))
                fout.write('\n')
