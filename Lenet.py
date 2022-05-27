import torch
import torchvision
import os
import argparse
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
from PIL import Image

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 參數設定
parser = argparse.ArgumentParser(description='PyTorch Lenet Example')
parser.add_argument('--no_download_data', action='store_true', default=False,
                    help='Do not download data')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--train', action='store_true', default=False,
                    help='Do not train')                    
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
#確定使用GPU

print('Use GPU:',args.cuda)

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

print("Number of epochs: ", args.epochs)
print("Batch size: ", args.batch_size)
#print("Log interval: ", args.log_interval)
print("Learning rate: ", args.lr)

train_data = torchvision.datasets.MNIST(
    root='./mnist',
    train=True,
    transform=torchvision.transforms.Compose([
                torchvision.transforms.Resize((32,32)),
                torchvision.transforms.ToTensor()
    ]),
    download=False
)
test_data = torchvision.datasets.MNIST(
    root='./mnist/', 
    train=False,
    transform=torchvision.transforms.Compose([
                torchvision.transforms.Resize((32,32)),
                torchvision.transforms.ToTensor()
    ]),
    download=False,
    )
#train_loader被分為(60000/batch_size)+1個batch 每個batch共有batch_size個data
train_loader = torch.utils.data.DataLoader(dataset = train_data, batch_size = args.batch_size, shuffle=True)
#test_loader被分為(10000/batch_size)+1個batch 每個batch共有batch_size個data
test_loader = torch.utils.data.DataLoader(dataset = test_data, batch_size = args.batch_size, shuffle=True)

#Lenet模型定義
class Lenet(nn.Module):
    def __init__(self):
        super(Lenet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(5*5*16, 120)  
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = torch.tanh(self.conv1(x))
        x = torch.max_pool2d(x,2,2)
        x = torch.tanh(self.conv2(x))
        x = F.max_pool2d(x,2,2)    
        x = x.view(-1,5*5*16)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = F.softmax(self.fc3(x),dim=1)
        return x

#使用GPU
model = Lenet()
if args.cuda:
    model.cuda()

#設定最佳化方法
optimizer = optim.SGD(model.parameters(),lr=args.lr,momentum=args.momentum)     

#定義訓練函數
def train(epoch):
    model.train()
    losses = []
    for batch_idx, (data, target) in enumerate(train_loader): #
        if args.cuda:
            data, target = data.cuda(), target.cuda()         #使用GPU   
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()                                 #清空上一次的梯度，若不清空，梯度會和上一個batch特徵有關
        output = model(data)                                  #前向傳播，計算各輸出機率
        loss = F.cross_entropy(output, target)                #計算損失函數
        loss.backward()                                       #反向傳播，計算梯度
        optimizer.step()                                      #基於梯度，更新參數
        if batch_idx % args.log_interval == 0:                #印出訓練資訊
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset),100. * batch_idx / len(train_loader), loss.item()))
 
    
#定義測試函數
def test(Model):
    Model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()  
        data, target = Variable(data), Variable(target)
        output = Model(data)                                 #模型預測各輸出之機率
        test_loss += F.cross_entropy(output, target).item()  #計算測試LOSS
        pred = output.data.max(1)[1]                         #獲得模型預測的機率最大值對應的數字索引 
        correct += pred.eq(target).cpu().sum()               #比對預測結果

    test_loss /= len(test_loader)                           #計算平均LOSS
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


if args.train :                                             #進行訓練
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(model)
    torch.save(model.state_dict(), "mnist_Lenet.pt")        #儲存模型參數

model2 = Lenet()                                            #用新的model確認儲存參數是否可用
if args.cuda:
    model2.cuda()
model2.load_state_dict(torch.load("mnist_Lenet.pt"))        #讀取儲存的參數
test(model2)                                                #新的model進行測式

ori_img = Image.open('3.png').convert('L')                  #透過小畫家自製手寫數字圖
t = torchvision.transforms.Compose([
    torchvision.transforms.Resize((32, 32)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.1307,), (0.3081,))
])
img = torch.autograd.Variable(t(ori_img).unsqueeze(0))
img = img.cuda()

model2.eval()
output = model2(img)                                        #輸出預測分布
pred = output.data.max(1)[1].item()                         #判定預測結果

plt.imshow(ori_img,cmap='gray')             
plt.title('Prediction: %i' % pred ,fontsize = 35)           
plt.show()                                                  #顯示數字圖與預測結果