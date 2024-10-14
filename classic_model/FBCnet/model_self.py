import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

class FBCNet(nn.Module):
    def __init__(self, m=32,C=30,Nc=2,T=200,Nb=9,w=50):
        super(FBCNet, self).__init__()
        self.dropout = 0.5
        self.conv1 = nn.Conv2d(Nb, m*Nb, (C, 1), padding=0)
        self.batchnorm1 = nn.BatchNorm2d(m*Nb, False)
        self.fc1 = nn.Linear(12607488,Nc)
        

    def forward(self, x,w=50):
        #Spatial convolution
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = F.elu(x)
        #Temporal Feature Extration
        x1 = torch.var(x[:,:,:,0:w], dim=3, keepdim=True)
        x2 = torch.var(x[:,:,:,w:2*w], dim=3, keepdim=True)
        x3 = torch.var(x[:,:,:,2*w:3*w], dim=3, keepdim=True)
        x4 = torch.var(x[:,:,:,3*w:4*w], dim=3, keepdim=True)
        x=torch.cat((x1,x2,x3,x4),dim=3)
        x = F.logsigmoid(x)
        print('log',x.shape)
        #   Classifier
        x = torch.flatten(x)
        print('faltten',x.shape)
        x = self.fc1(x)  # 使用线性层
        x = F.softmax(x)
        print(x.shape)
        return x

    def apply_max_norm(self, max_norm=2.0):
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm)

model=FBCNet()
# 测试代码
batch_size = 64  # 批量大小
Nb = 9           # 通道数
m = 32           # 乘数
C = 30           # 卷积核大小
T = 200          # 时间步长
w = 50           # 每个块的宽度

# 生成随机输入数据，形状为 (batch_size, Nb, T, w)
input_data = torch.randn(batch_size, Nb, T, w)

# 创建模型并进行前向传播
model = FBCNet()
outputs = model(input_data, w)

# 输出结果形状
print("输出形状:", outputs.shape)






