import torch.nn as nn

class DualHeadNN(nn.Module):
    def __init__(self, state_dim, k_min_dim, k_max_dim):
        super(DualHeadNN, self).__init__()
        self.shared_net = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.k_min_head = nn.Linear(32, k_min_dim)
        self.k_max_head = nn.Linear(32, k_max_dim)

    def forward(self, x):
        x = self.shared_net(x)
        return self.k_min_head(x), self.k_max_head(x)
    
# 三头神经网络（三个独立输出）
class TripleHeadNN(nn.Module):
    # 输入参数：状态维度、kmin维度、kmax维度、pmax维度
    def __init__(self, state_dim, k_min_dim, k_max_dim, p_max_dim):
        super(TripleHeadNN, self).__init__()
        # 共享网络：实现state的特征提取，将输入状态转化为32维的共享特征，分别经过全连接层->激活函数->全连接层...
        self.shared_net = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        # 三头输出层：简单的线性层
        self.k_min_head = nn.Linear(32, k_min_dim)
        self.k_max_head = nn.Linear(32, k_max_dim)
        self.p_max_head = nn.Linear(32, p_max_dim)
    # 向前传播函数：定义数据通过网络的计算流程 输入状态->得到32维度共享特征->得到三个输出头
    def forward(self, x):
        x = self.shared_net(x)
        return self.k_min_head(x), self.k_max_head(x), self.p_max_head(x)