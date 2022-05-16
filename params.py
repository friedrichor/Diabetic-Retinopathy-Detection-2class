import os
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

num_classes = 2

# model = ROOT / 'weights' / 'best_model.pth'  # 保存模型路径 & predict时调用的模型
model = ROOT / '../drive/MyDrive/weights_convnext' / 'best_model.pth'  # 保存模型路径 & predict时调用的模型
path_train = ROOT / 'data/train/'
path_test = ROOT / 'data/test/'
path_json = ROOT / 'class_indices.json'