import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

import torchvision.datasets as dsets
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np
import time

start = time.time()
# 定义超参数
image_size = 28     # 图片尺寸
num_classes = 10    # 分类标签
num_epochs = 20     # 循环次数
batch_size = 64     # 批处理大小

# 数据处理
train_dataset = dsets.MNIST(root='/data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = dsets.MNIST(root='/data', train=False, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

'''
idx = 110
mut = train_dataset[idx][0].numpy()
plt.imshow(mut[0,...])
plt.show()
print('标签是', train_dataset[idx][1])
'''
indices = range(len(test_dataset))
indices_val = indices[:5000]
indices_test = indices[5000:]
sampler_val = torch.utils.data.sampler.SubsetRandomSampler(indices_val)
sampler_test = torch.utils.data.sampler.SubsetRandomSampler(indices_test)
validation_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size,
                                             shuffle=False, sampler=sampler_val)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size,
                                             shuffle=False, sampler=sampler_test)
depth = [4, 8]
# 搭建网络
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, 5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(depth[0], depth[1], 5, padding=2)
        self.fc1 = nn.Linear(image_size//4*image_size//4*depth[1], 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)   # 第一层卷积
        x = F.relu(x)   # relu激活函数防止过拟合
        x = self.pool(x)    # 池化操作
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = x.view(-1, image_size//4*image_size//4*depth[1])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)    # dropout操作防止过拟合，速率为默认的0.5
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)     # 便于计算交叉熵
        return x

    def retrieve_features(self, x):
        feature_map1 = F.relu(self.conv1(x))    # 提取第一层卷积的特征图
        x = self.pool(feature_map1)
        feature_map2 = F.relu(self.conv2(x))
        return (feature_map1, feature_map2)

    # 模型运行
net = ConvNet()
criterion = nn.CrossEntropyLoss()   # 交叉熵
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
record = []
weights = []

def rightness(output, target):
    pred = torch.max(output.data, 1)[1]
    right = pred.eq(target.data.view_as(pred)).sum()
    return right, len(target)

for epoch in range(num_epochs):
    train_rights = []
    val_rights = []
    for batch_id, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        net.train()     # 开启dropout

        output = net(data)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        right = rightness(output, target)
        train_rights.append(right)

        if batch_id%100 == 0:
            net.eval()  # 关闭dropout
            for (data, target) in validation_loader:
                data, target = Variable(data), Variable(target)
                output = net(data)
                right = rightness(output, target)
                val_rights.append(right)
        train_num = (sum([tup[0] for tup in train_rights]), sum([tup[1] for tup in train_rights]))
        val_num = (sum([tup[0] for tup in val_rights]), sum([tup[1] for tup in val_rights]))
        print('批次：{} 训练准确率：{}% 验证准确率: {}%'.format(epoch, (100.0*train_num[0]/train_num[1]).numpy(),
                                                  (100.0*val_num[0]/val_num[1]).numpy()))
        record.append((100-100.*train_num[0]/train_num[1], 100-100.*val_num[0]/val_num[1]))
        weights.append([net.conv1.weight.data.clone(), net.conv1.bias.data.clone(),
                            net.conv2.weight.data.clone(), net.conv2.bias.data.clone()])

# 测试模型
net.eval()
vals = []
for (data, target) in test_loader:
    data, target = Variable(data), Variable(target)
    output = net(data)
    right = rightness(output, target)
    vals.append(right)
vals_num = (sum([tup[0] for tup in vals]), sum([tup[1] for tup in vals]))
rate = 100.0*vals_num[0]/vals_num[1]
print('测试精确度是: ', rate.numpy(), '%', '类型：', type(rate))
end = time.time()
print('程序运行时间为：', (end-start)/60.0, '分')

plt.figure(figsize=(10, 8))
plt.plot(record)
plt.xlabel('Steps')
plt.ylabel('error rate')
plt.show()








