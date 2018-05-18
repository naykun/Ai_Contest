# 环境配置
- python 3.5
- tensorflow 1.4及以上
- keras 2.1.4
- cuda 9.0
- cudnn 7.0
- matplotlib
- pytorch 0.4
- torchvision
- pretrainedmodels

# 训练数据准备
将datapre.py文件放置于如下结构中
```
.
│  categories.csv
│  list.csv
│  datapre.py
│  pic-copyrights.csv
└─data
        0039b0226653bac09d240194dbb6cdaf.jpg
        ....
        ffd354fc06a4a1561d2196fb31f877f5.jpg
```
执行命令：
```
python3 datapre.py
```    
数据集将被自动切分为训练集(./train_data)和验证集(./val_data)两个部分

# 训练模型

执行命令：
```
python3 train.py --train_dir {dir of train data} --val_dir {dir of validate data}
```  
模型将保存在./history目录下

# 验证模型

执行命令：
```
python3 validate.py --val_dir {dir of test data} --model {path of model to val} 
```

# 执行预测
将predict_batch.py 放置于如下目录结构中：
```
│  categories.csv
│  list.csv
│  predict_batch.py
│  pic-copyrights.csv
└─data
        0039b0226653bac09d240194dbb6cdaf.jpg
        ....
        ffd354fc06a4a1561d2196fb31f877f5.jpg
```

执行命令：
```
python3 predict_batch.py --model {path of model to predict} 
```
预测结果将以result_****.csv的形式保存在当前目录中