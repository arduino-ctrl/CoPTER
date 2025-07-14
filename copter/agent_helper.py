# Author: Tianyu Zuo (@amefumi)
# Email: ty.zuo@outlook.com

import numpy as np
import os
import random
from collections import deque
from loguru import logger

from structures import NetworkHelperParameters, DCQCNParameters, PortObservation, AgentParameters, AgentHelperParameters
from agent import Agent, ACC, CoPTER


DEFAULT_ACC_PARAMETER = AgentParameters(state_dim=18, kmin_dim=6, kmax_dim=4, pmax_dim=10, learning_rate=1e-3, gamma=0.95)
DEFAULT_COPTER_PARAMETER = AgentParameters(state_dim=18, kmin_dim=4, kmax_dim=6, pmax_dim=0, learning_rate=1e-3, gamma=0.95, kmin_res=40, kmax_res=60)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)


    def push(self, state, action, reward, next_state):
        self.buffer.append((state, action, reward, next_state))


    def push_batch(self, batch):
        for state, action, reward, next_state in batch:
            self.push(state, action, reward, next_state)


    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
        )


    def __len__(self):
        return len(self.buffer)
    


class AgentHelper:
    def __init__(
            self, 
            node_number: int, 
            ahp: AgentHelperParameters,
            model_dir: str,
            mode: str = "ACC",
            exp_name: str = "Default",
            online: bool = True,
            fmap_dir: str = None
            ):
        # Assert fmap_dir is provided if mode is CoPTER
        if mode == "CoPTER" and fmap_dir is None:
            raise ValueError("fmap_dir must be provided when mode is CoPTER.")

        # Initialize the AgentHelper with the given parameters
        logger.info(f"Initializing AgentHelper with mode: {mode}, node_number: {node_number}, online: {online}")
        
        self.node_number = node_number
        self.p = ahp
        self.model_dir = model_dir if model_dir is not None else "models"

        self.mode = mode
        self.exp_name = exp_name
        self.online = online

        # Load fmap if mode is CoPTER
        self.fmap = self._load_fmap(fmap_dir, 0) if mode == "CoPTER" else None

        # Initialize all agents and their replay buffers (if available)
        self.agent_pool: list[Agent] = []
        self.rb_pool = []
        for i in range(node_number):
            # Initialize agent by mode
            self.agent_pool.append(
                ACC(f"{self.exp_name}_ACC_{i}", DEFAULT_ACC_PARAMETER) if mode == "ACC" else 
                CoPTER(f"{self.exp_name}_CoPTER_{i}", DEFAULT_COPTER_PARAMETER, self.fmap, self.online)
                )
            self.rb_pool.append(ReplayBuffer(capacity=self.p.rb_size))

        # NOTE: Shared experience means agent periodically syncs its partial experience with a global replay buffer,
        # and samples partial experience from it to its own replay buffer.
        self.shared_rb = deque(maxlen=self.p.rb_size_global)

 
    def decide(self, port_states: list[list[float]], epsi=0.1) -> tuple[list[DCQCNParameters], list[tuple[int, int, int]]]:
        paras: list[DCQCNParameters] = []
        actions: list[tuple[int, int, int]] = []

        for port_idx, agent in enumerate(self.agent_pool):
            para, action = agent.select_action(port_states[port_idx], epsilon=epsi)
            paras.append(para)
            actions.append(action)
        
        return paras, actions
    

    def train(self, current_step: int):
        sample_size = self.p.train_set_size
        for port_idx, agent in enumerate(self.agent_pool):
            if len(self.rb_pool[port_idx]) > sample_size:
                states, actions, rewards, next_states = self.rb_pool[port_idx].sample(sample_size)

                logger.info(f"Training agent {agent.name} with {len(states)} samples.")
                agent.train_model(states, actions, rewards, next_states)

                if current_step % self.p.target_update_interval == 0:
                    logger.info(f"Updating target network for agent {agent.name} at step {current_step}.")
                    agent.update_target_network()
            else:
                logger.warning(f"Agent {agent.name} has insufficient experiences for training. Make sure to call sync() before training.")


    def load(self, override_name=None):
        # Load all agents' models from the model directory
        for agent in self.agent_pool:
            agent.load_model(self.model_dir, override_name)
            

        # # Load replay buffers if available
        # for i, rb in enumerate(self.rb_pool):
        #     rb_path = os.path.join(self.model_dir, f"{self.exp_name}_{self.agent_pool[i].name}_rb.npy")
        #     if os.path.exists(rb_path):
        #         rb_data = np.load(rb_path, allow_pickle=True)
        #         rb.push_batch(rb_data.tolist())
        #         logger.info(f"Loaded replay buffer for agent {self.agent_pool[i].name} from {rb_path}")
        #     else:
        #         logger.warning(f"No replay buffer found for agent {self.agent_pool[i].name} at {rb_path}")

        # # Load shared experience if available
        # shared_rb_path = os.path.join(self.model_dir, f"{self.exp_name}_shared_replay_buffer.npy")
        # if os.path.exists(shared_rb_path):
        #     shared_rb_data = np.load(shared_rb_path, allow_pickle=True)
        #     self.shared_rb.push_batch(shared_rb_data.tolist())
        #     logger.info(f"Loaded shared replay buffer from {shared_rb_path}")
        # else:
        #     logger.warning(f"No shared replay buffer found at {shared_rb_path}. Using empty shared replay buffer.")


    def save(self):
        # Save all agents' models from the model directory
        for agent in self.agent_pool:
            agent.save_model(self.model_dir)

        # # Save replay buffers if available
        # for i, rb in enumerate(self.rb_pool):
        #     rb_path = os.path.join(self.model_dir, f"{self.exp_name}_{self.agent_pool[i].name}_rb")
        #     np.save(rb_path, list(rb.buffer))
        #     logger.info(f"Saved replay buffer for agent {self.agent_pool[i].name} at {rb_path}")
            
        # # Save shared experience if available
        # shared_rb_path = os.path.join(self.model_dir, f"{self.exp_name}_shared_replay_buffer")
        # np.save(shared_rb_path, list(self.shared_rb.buffer))
        # logger.info(f"Saved shared replay buffer at {shared_rb_path}")


    def record(self, port_idx, state, action, reward, next_state):
        # Record a single experience into replay buffer
        if self.online:
            self.rb_pool[port_idx].push(state, action, reward, next_state)
            logger.info(f"Recorded experience for port {port_idx} recorded. Port experiences: {len(self.rb_pool[port_idx])}")
        else:
            logger.warning(f"AgentHelper is in offline mode. Experience recording is skipped for port {port_idx}.")
       

    def sync(self):
        # Sync up: Sample local replay buffer to shared replay buffer
        sync_up_size = min(len(self.rb_pool[0]), self.p.sync_up_size)
        for i, agent in enumerate(self.agent_pool):
            states, actions, rewards, next_states = self.rb_pool[i].sample(sync_up_size)
            self.shared_rb.extend(zip(states, actions, rewards, next_states))
            logger.info(f"Agent {agent.name} synced up {sync_up_size} experiences to shared replay buffer. Total shared experiences: {len(self.shared_rb)}")
        
        # Sync down: Sample shared replay buffer to local replay buffer
        sync_down_size = min(len(self.shared_rb), self.p.sync_down_size)
        for i, agent in enumerate(self.agent_pool):
            sampled_experiences = random.sample(self.shared_rb, sync_down_size)
            self.rb_pool[i].push_batch(sampled_experiences)
            logger.info(f"Agent {agent.name} synced down {sync_down_size} experiences from shared replay buffer. Local replay buffer size: {len(self.rb_pool[i])}")


    @staticmethod
    def _load_fmap(fmap_dir: str, node_idx: int) -> np.ndarray:
        fmap_row = DEFAULT_COPTER_PARAMETER.kmin_dim
        fmap_col = DEFAULT_COPTER_PARAMETER.kmax_dim

        fmap_path = os.path.join(fmap_dir, f"node_{node_idx}.fmap") if fmap_dir is not None else None

        if fmap_path is None or os.path.exists(fmap_path) is False:
            logger.warning("No existed fmap file. Using default fmaps.")
            return np.ones((fmap_row, fmap_col), dtype=float)
        else:
            fmap = np.loadtxt(fmap_path, dtype=float)
            if fmap.shape != (fmap_row, fmap_col):
                # If the shape of fmap is not correct, log a warning and return a default fmap
                logger.warning(f"fmap file {fmap_path} has wrong shape: {fmap.shape}. Expected shape: ({fmap_row}, {fmap_col}). Using default fmaps.")
                return np.ones((fmap_row, fmap_col), dtype=float)
            return fmap
    
