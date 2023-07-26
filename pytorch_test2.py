import torch
import math
import random
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)


class NeuralNetwork(nn.Module):
    def __init__(self, list_pre, list_layer):
        super(NeuralNetwork, self).__init__()
        self.layer_list = list_layer
        self.layer_num = 0
        self.activation_num = 0
        self.convolution_num = 0
        self.pool_num = 0

        self.sequential = self.layer_stack(self.layer_list)
        print(self.sequential)

    def forward(self, x):
        i = self.sequential(x)
        return i

    def layer_stack(self, x):
        print(x)
        seq = nn.Sequential()
        for i in x:
            if i[0] == 'ln':
                self.layer_num += 1
                seq.add_module('layer_' + str(self.layer_num) + '_Linear', nn.Linear(i[1], i[2]))
            elif i[0] == 'relu':
                self.activation_num += 1
                seq.add_module('activation_' + str(self.activation_num) + '_Relu', nn.ReLU())
            elif i[0] == 'sigmoid':
                self.activation_num += 1
                seq.add_module('activation_' + str(self.activation_num) + '_Sigmoid', nn.Sigmoid())

        return seq


def target_func(x1, x2, x3):
    return np.array(pow(1.59,x1))


def data_combine(x1, x2, x3, s, f):
    x = np.dstack((x1[s:f], x2[s:f], x3[s:f]))
    return x[0]


X1 = np.random.uniform(low=0, high=10, size=10000)
X2 = np.random.uniform(low=0, high=10, size=10000)
X3 = np.random.uniform(low=0, high=10, size=10000)

Y = target_func(X1, X2, X3)
train_size = 10000
val_size = 200
train_X = data_combine(X1, X2, X3, 0, train_size)
train_Y = Y[0:train_size]
train_Y = train_Y[:, np.newaxis]
wow = X1[:, np.newaxis]
#wow = np.transpose(wow)
print(wow)
print(wow.shape)
train_X=torch.from_numpy(wow).float().to(device)
train_Y=torch.from_numpy(train_Y).float().to(device)
print(train_X.shape)
print(train_Y)
list_pre = ['']

list_layer = [['ln', 1, 10],['relu'],['ln',10, 10],['relu'],['ln', 10, 1]]

model = NeuralNetwork(list_pre, list_layer).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# 학습
num_epochs = 10000

model.train()

for epoch in range(num_epochs):
    running_loss = 0.0

    optimizer.zero_grad()

    outputs = model(train_X)
    loss = criterion(outputs, train_Y)
    loss.backward()
    optimizer.step()

    running_loss += loss.item()
    if epoch%1000==0:
        print('Epoch: %d | Loss: %.4f' % (epoch + 1, running_loss))
model.eval()

test_input1 = np.linspace(0,10,10)
test_input2 = np.full(10,5)
test_input3 = np.full(10,1.5)

ti = np.dstack((test_input2[0:10], test_input1[0:10], test_input3[0:10]))
test_input = torch.from_numpy(test_input1[:, np.newaxis]).float().to(device)
output_data = model(test_input)
print(output_data)

# PyTorch 텐서를 NumPy 배열로 변환
output_data = output_data.detach().cpu().numpy()

# 그래프 그리기
plt.plot(test_input1, output_data, 'ro-')  # 입력 데이터와 출력 데이터를 그래프로 표현
plt.xlabel('Input')
plt.ylabel('Output')
plt.title('Model Prediction')
plt.show()


'''
for name, child in model.named_children():
    for param in child.parameters():
        print(name, param)
'''
