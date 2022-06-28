import torch
from torch import nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import module
from module import NeuralNetwork

inputTenserT = torch.rand(5, 3)
resultDataT = []
for line in inputTenserT:
    val = (line[0] * 2 + line[1] + line[2] * 2) / 3.0
    isBig = val > 0.5
    resultDataT.append([isBig])
resultTenserT = torch.tensor(resultDataT, dtype=float)



# To load a saved version of the model:
saved_model = NeuralNetwork()
saved_model.load_state_dict(torch.load(
    "基础测试，三个数字平均值是否大于0.5/model_20220628_134959_last"))
val0 = saved_model(torch.tensor(
    [[0, 0, 0], [1, 1, 1], [0.5, 0.5, 0.5], [0.49, 0.5, 0.5], [0.2, 0.2, 1.2], [0.49, 0.49, 0.49]], dtype=torch.float32))
val1 = saved_model(inputTenserT)
print(val0)
print(inputTenserT)
print(val1)
