import torch
import torchvision
from torchvision import datasets,transforms, models
import os
import numpy as np
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.autograd import Variable 
import time
import pretrainedmodels

parser = argparse.ArgumentParser(description='PyTorch Training')

parser.add_argument('data', metavar='DIR',
                    help='path to dataset')

parser.add_argument('-j', '--workers', default=6, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')

parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')

parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')

parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')

parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')

parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')

parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')

parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

parser.add_argument('--num_classes',default=20, type=int, help='num of class in the model')


use_gpu = torch.cuda.is_available()
print(use_gpu)
best_acc = 0.0

def exp_lr_scheduler(optimizer, epoch,  lr_decay_epoch=7):  
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""  
    lr = args.lr * (0.1**(epoch // lr_decay_epoch))  
  
    if epoch % lr_decay_epoch == 0:  
        print('LR is set to {}'.format(lr))  
  
    for param_group in optimizer.param_groups:  
        param_group['lr'] = lr  
  
    return optimizer  










def train_model(model,data_loader_image, criterion, optimizer, lr_scheduler):  
    
    
    since = time.time()  
  
    best_model = model  
  
    for epoch in range(args.start_epoch,args.epochs):  
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))  
        print('-' * 10)  
  
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        # Each epoch has a training and validation phase  
        for phase in ['train', 'val']:  
            if phase == 'train':  
                optimizer = lr_scheduler(optimizer, epoch)  
                model.train(True)  # Set model to training mode  
            else:  
                model.train(False)  # Set model to evaluate mode  
  
            running_loss = 0.0  
            running_corrects = 0  
            end = time.time()
            # Iterate over data.  
            for i,data in enumerate(data_loader_image[phase]):  
                data_time.update(time.time() - end)
                # get the inputs  
                inputs, labels = data  
  
                # wrap them in Variable  
                if use_gpu:  
                    inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())  
                else:  
                    inputs, labels = Variable(inputs), Variable(labels)  
  
                # zero the parameter gradients  
                optimizer.zero_grad()  
  
                # forward  
                outputs = model(inputs)  
                _, preds = torch.max(outputs.data, 1)  
                loss = criterion(outputs, labels)  

                # backward + optimize only if in training phase  
                if phase == 'train':  
                    loss.backward()  
                    optimizer.step()  
  
                # statistics
                # measure accuracy and record loss  
                prec1, prec5 = accuracy(output.data, target, topk=(1, 3))
                losses.update(loss.data[0], inputs.size(0))
                top1.update(prec1[0], inputs.size(0))
                top5.update(prec5[0], inputs.size(0))

                batch_time.update(time.time() - end)
                end = time.time()


                if i % args.print_freq == 0:
                print('{} Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    phase, epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1, top5=top5))
  
            
  
            print('{} * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.val:.4f} ({loss.avg:.4f})'
            .format(phrase,top1=top1, top5=top5,loss=losses)) 
  
            # deep copy the model  
            if phase == 'val' and top1.avg > best_acc:  
                best_acc = top1.avg  
                best_model = copy.deepcopy(model)
                torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc': best_acc,
                }, "model_nasnetalarge_finetune_latest.pkl")  
  
        print()  
    time_elapsed = time.time() - since  
    print('Training complete in {:.0f}m {:.0f}s'.format(  
        time_elapsed // 60, time_elapsed % 60))  
    print('Best val Acc: {:4f}'.format(best_acc))  
    return best_model



def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (inputs, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(inputs, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def main():
    global args, best_acc
    args = parser.parse_args()
    print args

    # create model
    print("=> creating model")
    model_name = 'nasnetalarge' 
    model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')

    #freeze part of the net
    for parma in model.parameters():
        parma.requires_grad = False

    dim_feats = model.last_linear.in_features 
    nb_classes = 20
    model.last_linear = nn.Linear(dim_feats, nb_classes)

    if use_gpu:
        model = torch.nn.DataParallel(model).cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    data_transforms = {  
        'train': transforms.Compose([  
            transforms.RandomResizedCrop(224), #从原图像随机切割一张（224， 224）的图像
            transforms.RandomHorizontalFlip(), #以0.5的概率水平翻转
            transforms.RandomVerticalFlip(), #以0.5的概率垂直翻转
            transforms.RandomRotation(10), #在（-10， 10）范围内旋转
            transforms.ColorJitter(0.05, 0.05, 0.05, 0.05), #HSV以及对比度变化
            transforms.ToTensor(), #把PIL.Image或者numpy.ndarray对象转化为tensor，并且是[0,1]范围，主要是除以255
            normalize  
        ]),  
        'val': transforms.Compose([  
            transforms.Scale(256),  
            transforms.CenterCrop(224),  
            transforms.ToTensor(),  
            normalize  
        ]),  
    }  

    path = args.data
    ifshuffle={'train':True,'val':False}
    data_image = {x:datasets.ImageFolder(root = os.path.join(path,x),
                                        transform = data_transform[x])
                for x in ["train", "val"]}

    data_loader_image = {x:torch.utils.data.DataLoader(dataset=data_image[x],
                                                    batch_size=args.batch_size, 
                                                    shuffle=ifshuffle[x],
                                                    num_workers=args.workers, 
                                                    pin_memory=True)
                        for x in ["train", "val"]}

    data_sizes = {x: len(data_image[x]) for x in ['train', 'val']}  
    data_classes = data_image['train'].classes

    class_to_idx = data_image['train'].class_to_idx 
    idx_to_class = dict(zip(data_image['train'].class_to_idx.values(), data_image['train'].class_to_idx.keys()))


    # define loss function (criterion) and pptimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), args.lr,
                                weight_decay=args.weight_decay)
    # optim.SGD([
    #                 {'params': model.base.parameters()},
    #                 {'params': model.classifier.parameters(), 'lr': 1e-3}
    #             ], lr=1e-2, momentum=0.9)

    if args.evaluate:
        validate(val_loader, model, criterion)
        return
    model = train_model(model,data_loader_image['train'], criterion, optimizer,exp_lr_scheduler)
    torch.save(model.state_dict(), "model_nasnetalarge_finetune.pkl")


if __name__ == '__main__':
    main()


