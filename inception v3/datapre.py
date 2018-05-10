import os
import shutil
from os import  walk
from os.path import join
import random

path = "./data"
train_data_path = "./train_data"
val_data_path = "./val_data"
# total num of files :16623
picname2cate = dict()
catelist = dict()

for i in range(20):
    catelist[str(i)] = []

with open("list.csv",'r') as fin:
    for iter,line in enumerate(fin):
        if iter != 0:
            line = line[:-1]
            picname2cate[line.split(',')[0]]=line.split(',')[1]
            catelist[line.split(',')[1]].append(picname2cate[line.split(',')[0]])
            # print(line.split(',')[0])


for i in range(20):
    print(str(i),":",len(catelist[str(i)]))
filenames = next(walk(path))[2]
num_files = len(filenames)
print(num_files)
print(filenames)

os.mkdir("train_data")
for i in range(20):
    os.makedirs(join("train_data",str(i)))


os.mkdir("val_data")
for i in range(20):
    os.makedirs(join("val_data",str(i)))

print(type(filenames))
random.seed(0)
random.shuffle(filenames)

for i,filename in enumerate(filenames):
    if i < 1600:
        shutil.copy(join(path,filename),join(val_data_path,picname2cate[filename[:-4]],filename))
    else:
        shutil.copy(join(path,filename),join(train_data_path,picname2cate[filename[:-4]],filename))
    print(filename)

# 0 : 783
# 1 : 556
# 2 : 810
# 3 : 666
# 4 : 1272
# 5 : 818
# 6 : 1336
# 7 : 1021
# 8 : 904
# 9 : 679
# 10 : 1536
# 11 : 215
# 12 : 663
# 13 : 596
# 14 : 867
# 15 : 664
# 16 : 587
# 17 : 1166
# 18 : 820
# 19 : 664
# 16623