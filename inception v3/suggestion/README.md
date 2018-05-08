修改建议：
1. Learning Rate需要Decay
2. 优化器建议选Adam
3. FineTune建议逐层打开
4. stop会在没找到最优解时关闭训练，这里需要吧patience加大
5. TensorBoard加一下
6. Evaluate文件根据我对train提出的建议修改
7. train.py和XcptionFineTune.py可以给你们一些参考

