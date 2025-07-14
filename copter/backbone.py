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
    

class TripleHeadNN(nn.Module):
    def __init__(self, state_dim, k_min_dim, k_max_dim, p_max_dim):
        super(TripleHeadNN, self).__init__()
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
        self.p_max_head = nn.Linear(32, p_max_dim)

    def forward(self, x):
        x = self.shared_net(x)
        return self.k_min_head(x), self.k_max_head(x), self.p_max_head(x)