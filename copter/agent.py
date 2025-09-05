import os
import numpy as np
import torch
import random
from loguru import logger
from scipy.interpolate import RegularGridInterpolator

from backbone import DualHeadNN, TripleHeadNN
from structures import DCQCNParameters, AgentParameters


class Agent:
    """
    Base class for agents.
    """
    def __init__(self):
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def save_model(self):
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def load_model(self):
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def select_action(self):
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def train_model(self):
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def update_target_network(self):
        raise NotImplementedError("This method should be implemented by subclasses.")


class ACC(Agent):
    """
    Tianyu's replicate of the ACC Agent for DCQCN from the original SIGCOMM paper.  
    You should follow the steps below to use this class properly:
    1. Initialize the agent with parameters.
    2. Call load_model() to load the pre-trained model if exists, or to initialize a new model.
    3. Use select_action() to get the action based on the current state.
    4. Call train_model() to train the model with the current state, action, reward, and next state.
    5. Call update_target_network() to update the target network priodically.
    6. Call save_model() to save the model to the specified path.
    """
    def __init__(self, name: str, agent_params: AgentParameters):
        # Agent Parameters Initialization
        self.p = agent_params
        self.name = name
        
        # Model Initialization
        self.device = torch.device("cpu")
        # 初始化策略网络（评估网络），每个训练步更新
        self.policy_net = TripleHeadNN(self.p.state_dim, self.p.kmin_dim, self.p.kmax_dim, self.p.pmax_dim).to(self.device)
        # 初始化目标网络，定期更新
        self.target_net = TripleHeadNN(self.p.state_dim, self.p.kmin_dim, self.p.kmax_dim, self.p.pmax_dim).to(self.device) 
        # 创建优化器，adam优化算法
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.p.learning_rate)
        # 计算损失函数
        self.loss_fn = torch.nn.MSELoss()


    def save_model(self, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(self.policy_net.state_dict(), os.path.join(save_path, f"{self.name}_policy.pt"))
        torch.save(self.target_net.state_dict(), os.path.join(save_path, f"{self.name}_target.pt"))
        logger.info(f"ACC Agent {self.name} - Model saved to {save_path}")

    
    def load_model(self, save_path, exp_name_override: str = None):
        parts = self.name.split("_ACC_", 1)
        if exp_name_override is not None and len(parts) == 2:
            name = f"{exp_name_override}_ACC_{parts[1]}"
        else:
            name = self.name

        policy_path = os.path.join(save_path, f"{name}_policy.pt")
        target_path = os.path.join(save_path, f"{name}_target.pt")

        if os.path.isfile(policy_path) and os.path.isfile(target_path):
            self.policy_net.load_state_dict(torch.load(policy_path))
            self.target_net.load_state_dict(torch.load(target_path))
            logger.info(f"ACC Agent {self.name} - Model loaded from {save_path}")
        else:
            logger.info(f"ACC Agent {self.name} - Model files not found in {save_path}, reinitializing models.")
            self.target_net.load_state_dict(self.policy_net.state_dict())

    # 基于贪心策略和基于state的策略网络来选择动作
    def select_action(self, state, epsilon=0.1) -> tuple[DCQCNParameters, tuple[int, int, int]]:
        """
        Select an action based on the current state and epsilon-greedy policy.
        Args:
            state (list): The current state of the environment.
            epsilon (float): The probability of selecting a random action.
        Returns:
            tuple: A tuple containing the selected action as a `DCQCNParameters` object and the action indices.
        """
        kmin_values = [0.0, 0.0949, 0.2259, 0.4066, 0.6560, 1.0]  # 6
        kmax_values = [0.0, 0.25, 0.5, 1.0]  # 4
        pmax_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # 10
        
        if random.random() < epsilon:
            action = (random.randint(0, self.p.kmin_dim - 1), 
                      random.randint(0, self.p.kmax_dim - 1), 
                      random.randint(0, self.p.pmax_dim - 1))
        else:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                kmin, kmax, pmax = self.policy_net(state)
                logger.info(f"ACC Agent {self.name} - Q Values: Kmin: {kmin}, Kmax: {kmax}, Pmax: {pmax}")
            kmin_index = kmin.argmax().item()
            kmax_index = kmax.argmax().item()
            pmax_index = pmax.argmax().item()
            action = (int(kmin_index), int(kmax_index), int(pmax_index))
        
        logger.info(f"ACC Agent {self.name} - Action: Kmin: {kmin_values[action[0]]}, Kmax: {kmax_values[action[1]]}, Pmax: {pmax_values[action[2]]}")
        # 返回kmin、kmax、pmax参数值以及索引值
        return (DCQCNParameters(kmin_values[action[0]], kmax_values[action[1]], pmax_values[action[2]]), action)
    
    
    def train_model(self, state, action, reward, next_state):
        logger.info(f"ACC Agent {self.name} - Training Start")

        # Format transformation from numpy to torch tensors
        state = torch.FloatTensor(state).to(self.device)
        action = torch.LongTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        
        # Q prediction value based on current state and action，需要进行梯度计算
        q_kmin_tensor, q_kmax_tensor, q_pmax_tensor = self.policy_net(state)  # torch.no_grad() couldn't be used here.
        
        # 拆分动作索引
        action_kmin_idx = action[:, 0]
        action_kmax_idx = action[:, 1]
        action_pmax_idx = action[:, 2]

        # 预测Q值汇总计算：Q = q_kmin + q_kmax + q_pmax
        q_prediction = (
            q_kmin_tensor.gather(1, action_kmin_idx.unsqueeze(1)) + 
            q_kmax_tensor.gather(1, action_kmax_idx.unsqueeze(1)) + 
            q_pmax_tensor.gather(1, action_pmax_idx.unsqueeze(1))
        )

        # Q value estimation based on reward and next state
        next_state = next_state.to(self.device)

        # 禁用梯度计算
        with torch.no_grad():
            # Double DQN
            # 用next_state 利用策略网络选择动作
            q_kmin_tensor, q_kmax_tensor, q_pmax_tensor = self.policy_net(next_state)
            action_kmin_idx = q_kmin_tensor.argmax(dim=1)
            action_kmax_idx = q_kmax_tensor.argmax(dim=1)
            action_pmax_idx = q_pmax_tensor.argmax(dim=1)
            # 用next_state 利用目标网络评估动作
            q_kmin_tensor, q_kmax_tensor, q_pmax_tensor = self.target_net(next_state)
            q_target = (
                q_kmin_tensor.gather(1, action_kmin_idx.unsqueeze(1)) +
                q_kmax_tensor.gather(1, action_kmax_idx.unsqueeze(1)) +
                q_pmax_tensor.gather(1, action_pmax_idx.unsqueeze(1))
            )
            # 目标网络Q值估计，原文中的yj=r+Q_target
            q_estimation = reward.unsqueeze(1).to(self.device) + self.p.gamma * q_target

        # Calculate the loss, perform backpropagation, and return the loss value for logging purposes.
        # 计算loss，利用均方误差
        loss = self.loss_fn(q_prediction, q_estimation)
        # 清空历史梯度
        self.optimizer.zero_grad()
        # 反向传播计算梯度
        loss.backward()
        # 更新网络参数，利用adam优化器
        self.optimizer.step()
        logger.info(f"ACC Agent {self.name} - Training End with Loss: {loss.item()}")

        return loss.item()


    def update_target_network(self):
        """
        Update the target network with the weights from the policy network.  
        Note that this method should be called by an Agent Helper periodically.
        """
        # 目标网络参数定期有策略网络参数覆盖
        self.target_net.load_state_dict(self.policy_net.state_dict())
        

class CoPTER(Agent):
    def __init__(self, name: str, agent_params: AgentParameters, f_matrix, online: bool = False):
        # Agent Parameters Initialization
        self.p = agent_params
        self.name = name
        self.online = online

        # Model Initialization
        self.device = torch.device("cpu")
        # copter修改为三头
        # self.policy_net = DualHeadNN(self.p.state_dim, self.p.kmin_dim, self.p.kmax_dim).to(self.device)
        # self.target_net = DualHeadNN(self.p.state_dim, self.p.kmin_dim, self.p.kmax_dim).to(self.device) 
        self.policy_net = TripleHeadNN(self.p.state_dim, self.p.kmin_dim, self.p.kmax_dim, self.p.pmax_dim).to(self.device)
        self.target_net = TripleHeadNN(self.p.state_dim, self.p.kmin_dim, self.p.kmax_dim, self.p.pmax_dim).to(self.device) 
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.p.learning_rate)
        self.loss_fn = torch.nn.MSELoss()

        # Guidance Function
        self.f_matrix = f_matrix


    def save_model(self, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(self.policy_net.state_dict(), os.path.join(save_path, f"{self.name}_policy.pt"))
        torch.save(self.target_net.state_dict(), os.path.join(save_path, f"{self.name}_target.pt"))
        logger.info(f"CoPTER Agent {self.name} model saved to {save_path}")

    
    def load_model(self, save_path, exp_name_override: str = None):
        parts = self.name.split("_ACC_", 1)
        if exp_name_override is not None and len(parts) == 2:
            name = f"{exp_name_override}_ACC_{parts[1]}"
        else:
            name = self.name

        policy_path = os.path.join(save_path, f"{name}_policy.pt")
        target_path = os.path.join(save_path, f"{name}_target.pt")
        if os.path.isfile(policy_path) and os.path.isfile(target_path):
            self.policy_net.load_state_dict(torch.load(policy_path))
            self.target_net.load_state_dict(torch.load(target_path))
            logger.info(f"CoPTER Agent {self.name} model loaded from {save_path}")
        else:
            logger.info(f"CoPTER Agent {self.name} model files not found in {save_path}, reinitializing models.")
            self.target_net.load_state_dict(self.policy_net.state_dict())


    # # This is doubao's help
    def select_action(self, state, epsilon=0.1) -> tuple[DCQCNParameters, tuple[int, int, int]]:
        """
        Select an action based on the current state and epsilon-greedy policy,
        with dynamic action space refinement guided by central performance matrix.
        Args:
            state (list): The current state of the environment.
            epsilon (float): The probability of selecting a random action.
        Returns:
            tuple: A tuple containing the selected action as a `DCQCNParameters` object and the action indices.
        """
        # 基础离散值（保持原维度不变）
        base_kmin = [0.0, 0.1429, 0.4286, 1.0]  # 4维：指数分布
        base_kmax = [0.0, 0.1667, 0.3333, 0.5, 0.75, 1.0]  # 6维：分段递增
        pmax_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # 10维
        
        # 在线模式-e-greedy
        if random.random() < epsilon and self.online:
            # 随机选择基础索引
            kmin_base_idx = random.randint(0, len(base_kmin) - 1)
            kmax_base_idx = random.randint(0, len(base_kmax) - 1)
            pmax_idx = random.randint(0, len(pmax_values) - 1)
            
            # # 从基础索引对应的细化值中随机选择
            # kmin_val = random.choice(kmin_mapping[kmin_base_idx])
            # kmax_val = random.choice(kmax_mapping[kmax_base_idx])
            
            # logger.info(f"CoPTER Agent {self.name} - Random Action: Kmin: {kmin_val}, Kmax: {kmax_val}, Pmax: {pmax_values[pmax_idx]}")
            # return (DCQCNParameters(kmin_valba, kmax_val, pmax_values[pmax_idx]), 
            #         (kmin_base_idx, kmax_base_idx, pmax_idx))
            return (DCQCNParameters(base_kmin[kmin_base_idx], base_kmax[kmax_base_idx], pmax_values[pmax_idx]), 
                    (kmin_base_idx, kmax_base_idx, pmax_idx))
        else:
            # 计算当前状态的Q值
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_kmin, q_kmax, q_pmax = self.policy_net(state)
            
            # 生成局部Q矩阵（4*6）
            q_matrix = q_kmin.cpu().numpy()[0][:, None] + q_kmax.cpu().numpy()[0][None, :]
            pmax_index = q_pmax.argmax().item()
            logger.info(f"CoPTER Agent {self.name} - Q Values: Kmin: {q_kmin}, Kmax: {q_kmax}, Pmax: {q_pmax}")
            
            # 全局指导矩阵（4*6）
            f_matrix = self.f_matrix.copy()
            
            # 1. 归一化处理
            def min_max_norm(matrix):
                """将矩阵归一化到[0,1]范围"""
                min_val = np.min(matrix)
                max_val = np.max(matrix)
                if max_val - min_val < 1e-6:  # 避免除以零
                    return np.zeros_like(matrix)
                return (matrix - min_val) / (max_val - min_val)
            
            q_norm = min_max_norm(q_matrix)  # 局部Q值归一化
            # q_norm = np.ones_like(q_matrix)  # 将 q_norm 设置为与 q_matrix 相同大小的全 1 矩阵
            # f_norm = min_max_norm(f_matrix)  # 全局矩阵归一化
            f_norm = f_matrix  # 全局矩阵不归一化，保持原始性能差异

            # --------------------------融合方案1--------------------------------------------------------------------
            # f_norm = np.ones_like(f_norm)  # 将 f_norm 设置为全 1 矩阵
            # fusion = q_norm*f_norm
            # fusion = q_norm
            # fusion = 1 * q_norm + 0 * f_norm  # 简单加权融合
            # fusion_index = np.unravel_index(np.argmax(fusion), fusion.shape)
            # fusion_kmin_value = base_kmin[fusion_index[0]]
            # fusion_kmax_value = base_kmax[fusion_index[1]]
            
            # # 日志输出
            # logger.info(f"CoPTER Agent {self.name} - Q Matrix:\n{q_matrix}\n"
            #             f"Guidance Matrix:\n{f_matrix}\n"
            #             f"Normalized Q Matrix:\n{np.array2string(q_norm, precision=4)}\n"
            #             f"Normalized Guidance Matrix:\n{np.array2string(f_norm, precision=4)}\n")
            #             # f"Best Base Index: {best_base_idx}\n"
            #             # f"Refined Kmin Candidates: {candidate_kmin}\n"
            #             # f"Refined Kmax Candidates: {candidate_kmax}")
            
            # # 构建返回结果
            # # fusion_index = (best_base_idx[0], best_base_idx[1], int(pmax_index))
            # # logger.info(f"CoPTER Agent {self.name} - Selected Action: Kmin: {best_kmin_val}, Kmax: {best_kmax_val}")
            # fusion_kmin_idx, fusion_kmax_idx = fusion_index  # 拆解二维元组
            # full_action_index = (fusion_kmin_idx, fusion_kmax_idx, pmax_index)  # 拼接为三维元组
            # return (DCQCNParameters(fusion_kmin_value, fusion_kmax_value, pmax_values[pmax_index]), 
            #         full_action_index)

            # --------------------------融合方案2----------------------------------------------------------------------
            # # 2. 获取Q矩阵前5个最大值的下标
            # # 展平矩阵并获取排序索引（降序）
            # q_flat = q_norm.flatten()
            # q_sorted_indices = np.argsort(q_flat)[::-1]  # 从大到小排序
            # # 取前5个并转换为二维索引
            # top_q_indices = [np.unravel_index(idx, q_norm.shape) for idx in q_sorted_indices[:5]]
            
            # # 3. 获取F矩阵前5个最大值的下标
            # f_flat = f_norm.flatten()
            # f_sorted_indices = np.argsort(f_flat)[::-1]  # 从大到小排序
            # # 取前5个并转换为二维索引
            # top_f_indices = [np.unravel_index(idx, f_norm.shape) for idx in f_sorted_indices[:5]]
            
            # # 4. 计算两个集合的交集
            # # 转换为可哈希的元组集合以便计算交集
            # top_q_set = set(tuple(idx) for idx in top_q_indices)
            # top_f_set = set(tuple(idx) for idx in top_f_indices)
            # intersection = top_q_set & top_f_set  # 交集
            
            # # 5. 确定最终选择的索引
            # if intersection:
            #     # 如果有交集，选择交集中F值最大的索引
            #     max_f_value = -np.inf
            #     best_index = None
            #     for idx in intersection:
            #         current_f_value = f_norm[idx]
            #         if current_f_value > max_f_value:
            #             max_f_value = current_f_value
            #             best_index = idx
            #     logger.info(f"选择了Q和F前5的交集: {best_index}，对应的F值: {max_f_value}")
            # else:
            #     # 如果没有交集，选择F矩阵中最大值的索引
            #     best_index = top_f_indices[0]  # 已经是排序后的第一个
            #     logger.info(f"Q和F前5没有交集，选择F矩阵最大值索引: {best_index}")
            
            # # 6. 获取最终选择的值
            # best_kmin_idx, best_kmax_idx = best_index
            # best_kmin_value = base_kmin[best_kmin_idx]
            # best_kmax_value = base_kmax[best_kmax_idx]
            
            
            # # 日志输出
            # logger.info(f"CoPTER Agent {self.name} - Q Matrix:\n{q_matrix}\n"
            #             f"Guidance Matrix:\n{f_matrix}\n"
            #             f"Normalized Q Matrix:\n{np.array2string(q_norm, precision=4)}\n"
            #             f"Normalized Guidance Matrix:\n{np.array2string(f_norm, precision=4)}\n"
            #             f"Q矩阵前5索引: {top_q_indices}\n"
            #             f"F矩阵前5索引: {top_f_indices}\n"
            #             f"交集: {list(intersection)}\n"
            #             f"最终选择索引: {best_index}")
            
            # # 构建返回结果
            # full_action_index = (best_kmin_idx, best_kmax_idx, pmax_index)
            # return (DCQCNParameters(best_kmin_value, best_kmax_value, pmax_values[pmax_index]), 
            #         full_action_index)
            # --------------------------融合方案3----------------------------------------------------------------
            # 2. 分别找到Q矩阵和F矩阵归一化后的最大值下标
            q_max_idx = np.unravel_index(np.argmax(q_norm), q_norm.shape)
            f_max_idx = np.unravel_index(np.argmax(f_norm), f_norm.shape)
            
            # 3. 融合策略：取行索引的最小值，列索引的最大值
            # 行索引（左端）取最小值
            fused_row = min(q_max_idx[0], f_max_idx[0])
            # 列索引（右端）取最大值
            fused_col = max(q_max_idx[1], f_max_idx[1])
            best_index = (fused_row, fused_col)
            
            # 日志输出融合过程
            logger.info(f"Q矩阵最大值下标: {q_max_idx}, F矩阵最大值下标: {f_max_idx}")
            logger.info(f"融合后下标: 行取最小({q_max_idx[0]}, {f_max_idx[0]})={fused_row}, 列取最大({q_max_idx[1]}, {f_max_idx[1]})={fused_col}")
            
            # 4. 获取最终选择的值
            best_kmin_idx, best_kmax_idx = best_index
            best_kmin_value = base_kmin[best_kmin_idx]
            best_kmax_value = base_kmax[best_kmax_idx]
            
            # --------------------------融合方案结束--------------------------
            
            # 日志输出
            logger.info(f"CoPTER Agent {self.name} - Q Matrix:\n{q_matrix}\n"
                        f"Guidance Matrix:\n{f_matrix}\n"
                        f"Normalized Q Matrix:\n{np.array2string(q_norm, precision=4)}\n"
                        f"Normalized Guidance Matrix:\n{np.array2string(f_norm, precision=4)}\n"
                        f"最终选择索引: {best_index}")
            
            # 构建返回结果
            full_action_index = (best_kmin_idx, best_kmax_idx, pmax_index)
            return (DCQCNParameters(best_kmin_value, best_kmax_value, pmax_values[pmax_index]), 
                    full_action_index)
    
    def train_model(self, state, action, reward, next_state):
        if self.online:
            logger.info(f"CoPTER Agent {self.name} - Online Training Start")

            # Format transformation from numpy to torch tensors
            state = torch.FloatTensor(state).to(self.device)
            action = torch.LongTensor(action).to(self.device)
            reward = torch.FloatTensor(reward).to(self.device)
            next_state = torch.FloatTensor(next_state).to(self.device)

            # Q value prediction based on current state and action
            state = state.to(self.device)
            q_kmin_tensor, q_kmax_tensor, q_pmax_tensor = self.policy_net(state)  # torch.no_grad() couldn't be used here.

            action_kmin_idx = action[:, 0]
            action_kmax_idx = action[:, 1]
            action_pmax_idx = action[:, 2]

            q_prediction = (
                q_kmin_tensor.gather(1, action_kmin_idx.unsqueeze(1)) + 
                q_kmax_tensor.gather(1, action_kmax_idx.unsqueeze(1)) +
                q_pmax_tensor.gather(1, action_pmax_idx.unsqueeze(1))
            )

            # Q value estimation based on reward and next state
            next_state = next_state.to(self.device)

            with torch.no_grad():
                q_kmin_tensor, q_kmax_tensor, q_pmax_tensor = self.policy_net(next_state)
                action_kmin_idx = q_kmin_tensor.argmax(dim=1)
                action_kmax_idx = q_kmax_tensor.argmax(dim=1)
                action_pmax_idx = q_pmax_tensor.argmax(dim=1)

                q_kmin_tensor, q_kmax_tensor, q_pmax_tensor = self.target_net(next_state)
                q_target = (
                    q_kmin_tensor.gather(1, action_kmin_idx.unsqueeze(1)) +
                    q_kmax_tensor.gather(1, action_kmax_idx.unsqueeze(1)) +
                    q_pmax_tensor.gather(1, action_pmax_idx.unsqueeze(1))
                )
                q_estimation = reward.unsqueeze(1).to(self.device) + self.p.gamma * q_target

            loss = self.loss_fn(q_prediction, q_estimation)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            logger.info(f"CoPTER Agent {self.name} - Online Training End with Loss: {loss.item()}")

            # We return the loss value for logging purposes.
            return loss.item()
        else:
            logger.info(f"CoPTER Agent {self.name} - Offline Training Skipped.")
            return 0.0


    def update_target_network(self):
        """
        Update the target network with the weights from the policy network.  
        Note that this method should be called by an Agent Helper through update(), which is periodically called by the main copter loop.
        """
        if self.online:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        else:
            logger.info(f"CoPTER Agent {self.name} - Target Network Update Skipped in Offline Mode.")


if __name__ == "__main__":
    state = np.ones(18)
    sample_f = np.array([[kmin_index + 0.1* kmax_index for kmax_index in range(6)] for kmin_index in range(4)])

    DEFAULT_ACC_PARAMETER = AgentParameters(state_dim=18, kmin_dim=6, kmax_dim=4, pmax_dim=10, learning_rate=1e-3, gamma=0.95)
    DEFAULT_COPTER_PARAMETER = AgentParameters(state_dim=18, kmin_dim=4, kmax_dim=6, pmax_dim=10, learning_rate=1e-3, gamma=0.95)

    agent = ACC("ACCAgent", DEFAULT_ACC_PARAMETER)
    agent.load_model("checkpoints")
    agent.select_action(state=state, epsilon=0.5)