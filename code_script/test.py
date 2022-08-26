# bert-base-chinese
# from transformers import AutoTokenizer, AutoModel
# pre_model = "nghuyong/ernie-health-zh"
# tokenizer = AutoTokenizer.from_pretrained(pre_model)
# model = AutoModel.from_pretrained(pre_model)
# inputs = tokenizer("Hello world! 苏 志 同", return_tensors="pt")
# outputs = model(**inputs)
# print(outputs)

import datetime
start_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print("start_time: " + str(start_time))

import synonyms
print("部位: ", synonyms.nearby("部位",20))
print("疾病: ", synonyms.nearby("疾病",20))
print("手术: ", synonyms.nearby("手术",20))
print("药品: ", synonyms.nearby("药品",20))

print("NOT_EXIST: ", synonyms.nearby("NOT_EXIST"))

complate_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print("complate_time: " + str(complate_time))

#计算两个日期的间隔
print("time spend: ",datetime.datetime.strptime(complate_time, '%Y-%m-%d %H:%M:%S') - datetime.datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S'))

# d1 = datetime.datetime.strptime('2012-03-05 17:41:20', '%Y-%m-%d %H:%M:%S')
# d2 = datetime.datetime.strptime('2012-03-02 17:41:20', '%Y-%m-%d %H:%M:%S')


# no_re_list=[{'a':'1','b':'2','c':'3'},{'a':'1','b':'2','c':'3'},{'a':'1','b':'2','c':'2'},{'a':'1','b':'2','c':'4'}]
#
# unique_list = []
# temp_no_re_list = sorted(no_re_list, key=lambda x: x['a'])
# for n in range(len(temp_no_re_list)):
#     if temp_no_re_list[n] not in unique_list:
#         unique_list.append(temp_no_re_list[n])
# no_re_list = unique_list
# print(no_re_list)

# # 导入库
# import matplotlib.pyplot as plt
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import random_split
# from torch.nn import functional as F
# import torchvision
# from torchvision import datasets,transforms
# import torchvision.transforms as transforms
# import optuna
# import os
#
# # 导入数据集
# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# CLASSES = 10
# DIR = os.getcwd()
# EPOCHS = 10
# train_dataset = torchvision.datasets.MNIST('classifier_data', train=True, download=True)
# m=len(train_dataset)
# transform = torchvision.transforms.Compose([    torchvision.transforms.ToTensor()])
# train_dataset.transform=transform
#
# # 将卷积神经网络一起定义到要调整的超参数
# # 在Optuna中，目标是最小化/最大化目标函数，它将一组超参数作为输入并返回验证分数。对于每个超参数，需要考虑不同范围的值。
# # 优化的过程称为研究，而对目标函数的每次评估称为试验。“Suggest API”在模型架构内被调用，为每个试验动态生成超参数。
# # 可以定义超参数范围的函数：
# # suggest_int 建议为第二完全连接层的输入单元设置整数值
# # suggest_float 建议dropout率的浮点值，在第二个卷积层（0-0.5，步长为0.1）和第一个线性层（0-0.3，步长为0.1）之后作为超参数引入。
# # suggest_categorical 建议优化器的分类值，稍后将显示
# class ConvNet(nn.Module):
#   def __init__(self, trial):
#     # We optimize dropout rate in a convolutional neural network.
#     super(ConvNet, self).__init__()
#     self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
#     self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
#     dropout_rate = trial.suggest_float("dropout_rate", 0, 0.5,step=0.1)
#     self.drop1=nn.Dropout2d(p=dropout_rate)
#     fc2_input_dim = trial.suggest_int("fc2_input_dim", 32, 128,32)
#     self.fc1 = nn.Linear(32 * 7 * 7, fc2_input_dim)
#     dropout_rate2 = trial.suggest_float("dropout_rate2", 0, 0.3,step=0.1)
#     self.drop2=nn.Dropout2d(p=dropout_rate2)
#     self.fc2 = nn.Linear(fc2_input_dim, 10)
#     def forward(self, x):
#       x = F.relu(F.max_pool2d(self.conv1(x),kernel_size = 2))
#       x = F.relu(F.max_pool2d(self.conv2(x),kernel_size = 2))
#       x = self.drop1(x)
#       x = x.view(x.size(0),-1)
#       x = F.relu(self.fc1(x))
#       x = self.drop2(x)
#       x = self.fc2(x)
#       return x
#
# # 定义了一个函数来尝试训练集中batch_size的不同值。它将训练数据集和批大小作为输入(稍后将在目标函数中定义)，并返回训练和验证加载器对象。
# def get_mnist(train_dataset, batch_size):
#   train_data, val_data = random_split(train_dataset, [int(m - m * 0.2), int(m * 0.2)])
#   # The dataloaders handle shuffling, batching, etc...
#   train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
#   valid_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)
#   return train_loader, valid_loader
#
# # 最重要的一步是定义目标函数，它使用采样程序来选择每次试验中的超参数值，并返回在该试验中获得的验证准确度。
# def objective(trial):
#   # Generate the model.
#   model = ConvNet(trial).to(DEVICE)
#   # Generate the optimizers.
#   # try RMSprop and SGD
#   '''
#   optimizer_name = trial.suggest_categorical("optimizer", ["RMSprop", "SGD"])
#   momentum = trial.suggest_float("momentum", 0.0, 1.0)
#   lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
#   optimizer = getattr(optim, optimizer_name)
#   (model.parameters(), lr=lr,momentum=momentum)
#   '''
#   #try Adam, AdaDelta adn Adagrad
#   optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "Adadelta","Adagrad"])
#   lr = trial.suggest_float("lr", 1e-5, 1e-1,log=True)
#   optimizer = getattr(optim, optimizer_name)
#   (model.parameters(), lr=lr)
#   batch_size=trial.suggest_int("batch_size", 64, 256,step=64)
#   criterion=nn.CrossEntropyLoss()
#   # Get the MNIST imagesset.
#   train_loader, valid_loader = get_mnist(train_dataset,batch_size)
#   # Training of the model.
#   for epoch in range(EPOCHS):
#     model.train()
#     for batch_idx, (images, labels) in enumerate(train_loader):
#       # Limiting training images for faster epochs.
#       #if batch_idx * BATCHSIZE >= N_TRAIN_EXAMPLES:
#       #    break
#       images, labels = images.to(DEVICE), labels.to(DEVICE)
#       optimizer.zero_grad()
#       output = model(images)
#       loss = criterion(output, labels)
#       loss.backward()
#       optimizer.step()
#       # Validation of the model.
#       model.eval()
#       correct = 0
#       with torch.no_grad():
#         for batch_idx, (images, labels) in enumerate(valid_loader):
#           # Limiting validation images.
#           # if batch_idx * BATCHSIZE >= N_VALID_EXAMPLES:
#           #    break
#           images, labels = images.to(DEVICE), labels.to(DEVICE)
#           output = model(images)
#           # Get the index of the max log-probability.
#           pred = output.argmax(dim=1, keepdim=True)
#           correct += pred.eq(labels.view_as(pred)).sum().item()
#           accuracy = correct / len(valid_loader.dataset)
#           trial.report(accuracy, epoch)
#           # Handle pruning based on the intermediate value.
#           if trial.should_prune():
#             raise optuna.exceptions.TrialPruned()
#             return accuracy
#
# # 下面创建一个study对象来最大化目标函数，然后使用study.optimize(objective,n_trials = 20)进行研究,
# # 将试验次数定为20次。可以根据问题的复杂程度对其进行更改
# study = optuna.create_study(direction='maximize')
# study.optimize(objective, n_trials=20)
# trial = study.best_trialprint('Accuracy: {}'.format(trial.value))
# print("Best hyperparameters: {}".format(trial.params))
#
# # 为了更容易地可视化最近5次试验中选择的超参数，我们可以构建一个DataFrame对象:
# df = study.trials_dataframe().drop(['state','datetime_start','datetime_complete','duration','number'], axis=1)
# df.tail(5)