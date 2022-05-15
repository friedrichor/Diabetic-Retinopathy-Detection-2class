# Diabetic-Retinopathy-Detection-2class
任务：二分类的Diabetic Retinopathy Detection（0-1归为一类，2-3归为一类）

使用ConvNeXt model

1. 每个文件中通用的参数在params.py
2. divide_datasets.py 为将原始的数据集按类划分，并划分为训练集（训练集+验证集）和测试集。  
原始数据集在'data_ori/datasets/'，其相应的类别信息在'data_ori/Mess1_annotation_train.csv'  
按类划分后的数据集保存在'data_ori/datasets_div/'，即'data_ori/datasets_div/0-1/'中为0-1类，'data_ori/datasets_div/2-3/'中为2-3类  
划分训练集和测试集以9:1划分，且是按类划分的（即两类的测试集数量并不一致，尚未知这样划分的测试集是否可靠），训练集保存在'data_ori/train/',测试集保存在'data_ori/test/'
3. img_process.py 为扩充训练集  
先通过随机翻转（当前只考虑水平翻转，翻转哪张图片是随机取得）将两类训练集扩充到相同数目（这里取600，可修改）  
然后将每张图片随机增强亮度、对比度、色度、锐度，扩充到原来的5倍（即最后训练集每类均有3k张）
4. 在train.py 中，可以修改预训练参数，在--weights中，如果修改的话要在文件开头中 from model import convnext_tiny as create_model 修改其中的convnext_tiny部分，与预训练模型相对应。同理，predict.py的import部分也要更改成相应函数。



PS：此外也尝试了EfficientNetV2，30轮效果训练集和验证集不到90%，效果并不好，因此没有继续尝试下去。
