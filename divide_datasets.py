import os
import shutil
import random
from tqdm import tqdm
import pandas as pd


# 按类划分数据集
def datasets_div(file, content_ori, content_div):  # csv文件，原始数据集目录路径，划分后数据集目录路径
    df = pd.read_csv(file)
    df = df[['image', 'Retinopathy_grade']]
    # 按类创建处理后的图片文件夹
    for name in ['0-1', '2-3']:
        folder = content_div + str(name)
        if os.path.exists(folder):
            shutil.rmtree(folder)  # 文件夹存在则删除(清空文件的目的)
        os.makedirs(folder)
    # 把grade=0-1的分到一个文件夹，grade=2-3的分到一个文件夹
    for index, row in df.iterrows():
        img, grade = row[0] + '.png', row[1]
        if (grade == 0) or (grade == 1):
            shutil.copy(content_ori + img, content_div + '0-1/')
        else:
            shutil.copy(content_ori + img, content_div + '2-3/')
    print('图片总量 =', len(os.listdir(content_ori)))
    print('grade=0-1图片数量 =', len(os.listdir(content_div + '0-1/')))
    print('grade=2-3图片数量 =', len(os.listdir(content_div + '2-3/')))


# 划分数据集，分为两类：(1)训练集+验证集;(2)测试集
def split_train_test(content_div, content_split, rate_test=0.1):
    for cls in os.listdir(content_div):
        content_cls = content_div + cls + '/'  # 当前类别的目录路径
        num_cls = len(os.listdir(content_cls))  # 当前类别中图片数量
        test_set = random.sample(os.listdir(content_cls), int(num_cls*rate_test))  # 随机取 num_cls*rate_test 个图片作为测试集

        content_train_cls = content_split + 'train/' + cls + '/'
        content_test_cls = content_split + 'test/' + cls + '/'
        if os.path.exists(content_train_cls):  # 清空
            shutil.rmtree(content_train_cls)
        os.makedirs(content_train_cls)
        if os.path.exists(content_test_cls):  # 清空
            shutil.rmtree(content_test_cls)
        os.makedirs(content_test_cls)

        print('正在划分数据集...')
        for img in tqdm(os.listdir(content_cls)):
            if img in test_set:
                shutil.copy(content_cls + img, content_test_cls + img)
            else:
                shutil.copy(content_cls + img, content_train_cls + img)


if __name__ == '__main__':
    file = 'data/Mess1_annotation_train.csv'
    content_ori = 'data/train/'
    content_div = 'data/train_div/'
    content_split = 'data/data_split/'
    for content in [content_div, content_split]:
        if not os.path.exists(content):
            os.mkdir(content)
    datasets_div(file, content_ori, content_div)
    split_train_test(content_div, content_split)


