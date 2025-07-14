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
        self.policy_net = TripleHeadNN(self.p.state_dim, self.p.kmin_dim, self.p.kmax_dim, self.p.pmax_dim).to(self.device)
        self.target_net = TripleHeadNN(self.p.state_dim, self.p.kmin_dim, self.p.kmax_dim, self.p.pmax_dim).to(self.device) 
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.p.learning_rate)
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
            kmin_index = kmin.argmax().item()
            kmax_index = kmax.argmax().item()
            pmax_index = pmax.argmax().item()
            action = (int(kmin_index), int(kmax_index), int(pmax_index))
        
        logger.info(f"ACC Agent {self.name} - Action: Kmin: {kmin_values[action[0]]}, Kmax: {kmax_values[action[1]]}, Pmax: {pmax_values[action[2]]}")
        return (DCQCNParameters(kmin_values[action[0]], kmax_values[action[1]], pmax_values[action[2]]), action)
    
    
    def train_model(self, state, action, reward, next_state):
        logger.info(f"ACC Agent {self.name} - Training Start")

        # Format transformation from numpy to torch tensors
        state = torch.FloatTensor(state).to(self.device)
        action = torch.LongTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        
        # Q prediction value based on current state and action
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

        # Calculate the loss, perform backpropagation, and return the loss value for logging purposes.
        loss = self.loss_fn(q_prediction, q_estimation)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        logger.info(f"ACC Agent {self.name} - Training End with Loss: {loss.item()}")

        return loss.item()


    def update_target_network(self):
        """
        Update the target network with the weights from the policy network.  
        Note that this method should be called by an Agent Helper periodically.
        """
        self.target_net.load_state_dict(self.policy_net.state_dict())
        

class CoPTER(Agent):
    def __init__(self, name: str, agent_params: AgentParameters, f_matrix, online: bool = False):
        # Agent Parameters Initialization
        self.p = agent_params
        self.name = name
        self.online = online

        # Model Initialization
        self.device = torch.device("cpu")
        self.policy_net = DualHeadNN(self.p.state_dim, self.p.kmin_dim, self.p.kmax_dim).to(self.device)
        self.target_net = DualHeadNN(self.p.state_dim, self.p.kmin_dim, self.p.kmax_dim).to(self.device) 
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


    def select_action(self, state, epsilon=0.1) -> tuple[DCQCNParameters, tuple[int, int, int]]:
        """
        Select an action based on the current state and epsilon-greedy policy.
        Args:
            state (list): The current state of the environment.
            epsilon (float): The probability of selecting a random action.
            online (bool): If True, use online learning (discret actions); otherwise, use offline learning (continous actions).
            kmin_res (int): The resolution for kmin values in offline inference.
            kmax_res (int): The resolution for kmax values in offline inference.
        Returns:
            tuple: A tuple containing the selected action as a `DCQCNParameters` object and the action indices.
        """
        kmin_values = [0.0, 0.3334, 0.6667, 1.0]
        kmax_values = [0.0, 0.2001, 0.4001, 0.6001, 0.8001, 1.0]  # 6
        
        if random.random() < epsilon and self.online:
            random_action = (random.randint(0, self.p.kmin_dim - 1),
                      random.randint(0, self.p.kmax_dim - 1))
            logger.info(f"CoPTER Agent {self.name} - Random Action: Kmin: {kmin_values[random_action[0]]}, Kmax: {kmax_values[random_action[1]]}")
            return (DCQCNParameters(kmin_values[random_action[0]], kmax_values[random_action[1]], 0.0), random_action)
        else:
            # Calculate the Q values for the current state
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_kmin, q_kmax = self.policy_net(state)

            # Fuse the Q values with the guidance matrix
            q_matrix = q_kmin.cpu().numpy()[0][:, None] + q_kmax.cpu().numpy()[0][None, :]

            fusion = q_matrix * self.f_matrix
            # TODO: Currently, we take f_matrix into cosideration for online learning (to better learn the desired action?), 
            #       but we shall remove it in offline learning in the future.
            
            fusion_index = np.unravel_index(np.argmax(fusion), fusion.shape)
            fusion_kmin_value = kmin_values[fusion_index[0]]
            fusion_kmax_value = kmax_values[fusion_index[1]]

            # print(f"Q Fusion Shape: {fusion.shape}, Simple Fusion Action: {fusion_index} = {fusion_kmin_value}, {fusion_kmax_value}. Q Fusion:\n{fusion}")

            if not self.online:
                # Interploate the Fusion-Q values to find the possibly optimal action
                fusion_interp = RegularGridInterpolator((kmin_values, kmax_values), fusion, method='cubic')

                # Search maximum and argmax of the interpolated Q values
                kmin_values_interp = np.linspace(0, 1, self.p.kmin_res)
                kmax_values_interp = np.linspace(0, 1, self.p.kmax_res)
                kmin_grid, kmax_grid = np.meshgrid(kmin_values_interp, kmax_values_interp, indexing='ij')
                fusion_values_interp = fusion_interp((kmin_grid, kmax_grid))
                # print(f"Interpolated Q Fusion Shape: {fusion_values_interp.shape}, Interpolated Q Fusion:\n{fusion_values_interp}")
                
                # Find the maximum value and its index
                # TODO: Use scipy.optimize to find the maximum of the interpolated Q values
                # res = minimize(lambda x: -q_fusion_interp(x), start_point, bounds=[(0.001, 0.999), (0.001, 0.999)])
                
                fusion_index_interp = np.unravel_index(np.argmax(fusion_values_interp), fusion_values_interp.shape)
                fusion_kmin_value = kmin_values_interp[fusion_index_interp[0]]
                fusion_kmax_value = kmax_values_interp[fusion_index_interp[1]]
                # print(f"Max Interpolated Q Fusion Value: {fusion_values_interp[fusion_index_interp]}, at Kmin: {fusion_kmin_value}, Kmax: {fusion_kmax_value}")
            
            logger.info(f"CoPTER Agent {self.name} - Selected Action: Kmin: {fusion_kmin_value}, Kmax: {fusion_kmax_value}")
            return (DCQCNParameters(fusion_kmin_value, fusion_kmax_value), fusion_index)

    
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
            q_kmin_tensor, q_kmax_tensor = self.policy_net(state)  # torch.no_grad() couldn't be used here.

            action_kmin_idx = action[:, 0]
            action_kmax_idx = action[:, 1]

            q_prediction = (
                q_kmin_tensor.gather(1, action_kmin_idx.unsqueeze(1)) + 
                q_kmax_tensor.gather(1, action_kmax_idx.unsqueeze(1))
            )

            # Q value estimation based on reward and next state
            next_state = next_state.to(self.device)

            with torch.no_grad():
                q_kmin_tensor, q_kmax_tensor = self.policy_net(next_state)
                action_kmin_idx = q_kmin_tensor.argmax(dim=1)
                action_kmax_idx = q_kmax_tensor.argmax(dim=1)

                q_kmin_tensor, q_kmax_tensor = self.target_net(next_state)
                q_target = (
                    q_kmin_tensor.gather(1, action_kmin_idx.unsqueeze(1)) +
                    q_kmax_tensor.gather(1, action_kmax_idx.unsqueeze(1))
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
    DEFAULT_COPTER_PARAMETER = AgentParameters(state_dim=18, kmin_dim=4, kmax_dim=6, pmax_dim=0, learning_rate=1e-3, gamma=0.95)

    agent = ACC("ACCAgent", DEFAULT_ACC_PARAMETER)
    agent.load_model("checkpoints")
    agent.select_action(state=state, epsilon=0.5)