# Author: Tianyu Zuo (@amefumi)
# Email: ty.zuo@outlook.com

from ns3gym import ns3env
import numpy as np
import json
from collections import deque
from loguru import logger
from structures import NetworkHelperParameters, DCQCNParameters, PortObservation


class NetworkHelper:
    def __init__(self, ns3_socket, nhp: NetworkHelperParameters):
        # Store the NetworkHelperParameters
        self.nhp = nhp
        logger.info(f"NetworkHelper initialized with parameters: {self.nhp}")
        
        # Initialize ns3-gym environment
        self.env = ns3env.Ns3Env(port=ns3_socket, startSim=False)
        logger.info("Waiting for ns3 simulator to start...")
        self.env.reset()

        # Observation and action space
        self.ob_space = self.env.observation_space
        self.ac_space = self.env.action_space
        logger.info(f"Observation space: {self.ob_space} -> shape = {self.ac_space.shape}")
        logger.info(f"Action space: {self.ac_space} -> shape = {self.ac_space.shape}")

        # Switch ports (equals to numbers of Agents)
        self.n_port = self.ob_space.shape[0] // nhp.port_states
        logger.info(f"Number of switch ports: {self.n_port}")

        # Observation history as `State`
        self.obs_history = [deque(maxlen=nhp.state_observations + 1) for _ in range(self.n_port)]
        logger.info(f"Observation history sizes: {len(self.obs_history)}")

        # Initialize an empty action for next step
        self.action = [0.0] * (self.n_port * self.nhp.port_actions)   
        self.action_port_bitmap = [0] * self.n_port  # Track the port index for each action

        # Port identifier map for logging and analysis
        self.port_identifier_map = {}  # port_idx → "switch_id-connected_node_id"


    def close_env(self):
        self.env.close()
        logger.info("ns3-gym environment closed.")

    # 为指定端口设置动作
    def configurator(self, curr_step, port_idx, port_action: DCQCNParameters):
        """
        Insert the action of a single port into the action list in accordance with the port index.
        Input Definition:
        - `port_idx`: The index of the port (0 to n_port-1).
        - `port_action`: A DCQCNParameters for the specified port.
        """

        if port_idx < 0 or port_idx >= self.n_port:
            raise ValueError(f"Invalid port index: {port_idx}. Must be between 0 and {self.n_port - 1}.")
        
        start_index = port_idx * self.nhp.port_actions
        end_index = start_index + self.nhp.port_actions
        self.action[start_index:end_index] = port_action.k_min_norm, port_action.k_max_norm, port_action.p_max
        logger.info(f"Step {curr_step} - Port {port_idx} - Action set to {port_action}.")
        self.action_port_bitmap[port_idx] = 1  # Mark the port as having an action set

    # 执行动作并获取新state
    def monitor(self, curr_step):
        """
        Step actions, monitor the environments, and save the observation history.
        Input Definition:
        - `curr_step`: The current step in the simulation (0 for initial observation, >0 for action).
        """

        if curr_step == 0:
            # If current step is 0, reset the environment and get the initial observation
            obs = self.env.reset()
            obs = np.array(obs)
            # done = False
            # 2. 发送空动作（需确保所有端口标记为已设置动作）
            default_param = DCQCNParameters()  # 默认值：k_min_norm=1.0, k_max_norm=1.0, p_max=0.2
            logger.info(f"Step 0 - 使用默认参数初始化动作，而非空动作: {default_param}")
            
            # 2. 为所有端口填充默认动作（每个端口占3个位置：kmin/kmax/pmax）
            self.action = []  # 清空原有全0列表
            for port_idx in range(self.n_port):
                # 依次添加当前端口的3个动作参数
                self.action.extend([
                    default_param.k_min_norm,
                    default_param.k_max_norm,
                    default_param.p_max
                ])
            self.action_port_bitmap = [1] * self.n_port  # 标记所有端口已设置（空动作）
            
            # 3. 首次step，获取环境信息（含port_identifiers）
            obs, _, done, info = self.env.step(self.action)
            logger.info(f"NS3 返回的 info 类型: {type (info)}") # 查看是 dict 还是 str
            logger.info(f"NS3 返回的 info 完整内容: {info}") # 打印完整信息
            # 关键：将字符串info解析为JSON字典
            parsed_info = {}
            try:
                # 尝试解析JSON字符串
                parsed_info = json.loads(info)
                logger.info(f"成功解析info为JSON: {parsed_info}")
            except json.JSONDecodeError as e:
                logger.error(f"info解析失败: {e}，原始内容: {info}")
                parsed_info = {"raw_info": info}
            # 处理port_identifiers
            if 'port_identifiers' in parsed_info:
                received_identifiers = parsed_info['port_identifiers']
                # 验证标识数量与端口数一致（448个）
                if len(received_identifiers) == self.n_port:
                    self.port_identifier_map = {
                        idx: ident for idx, ident in enumerate(received_identifiers)
                    }
                    logger.success(f"✅ 成功获取{len(received_identifiers)}个端口标识（与端口数{self.n_port}匹配）")
                    logger.info(f"端口标识映射表（前5个）: {dict(list(self.port_identifier_map.items())[:5])}")
                else:
                    logger.error(f"❌ 端口标识数量不匹配！接收{len(received_identifiers)}个，预期{self.n_port}个")
            else:
                logger.warning(f"当前info无port_identifiers（可能是偶发延迟），内容: {parsed_info}")
                # 重试一次（避免偶发通信问题）
                obs, _, done, retry_info = self.env.step(self.action)
                try:
                    retry_parsed = json.loads(retry_info)
                    if 'port_identifiers' in retry_parsed:
                        self.port_identifier_map = {
                            idx: ident for idx, ident in enumerate(retry_parsed['port_identifiers'])
                        }
                        logger.success(f"🔄 重试后获取到{len(retry_parsed['port_identifiers'])}个端口标识")
                except json.JSONDecodeError:
                    logger.error(f"🔄 重试解析失败: {retry_info}")
            obs = np.array(obs)
        else:
            # 验证所有端口都设置了动作->执行动作->重置动作位图
            assert sum(self.action_port_bitmap) > self.n_port - 1, "Not all ports have actions set. Please check the configurator."
            obs, _, done, info = self.env.step(self.action)
            # logger.info(obs)
            print(obs)
            obs = np.array(obs)
            # print(self.action)    
            self.action_port_bitmap = [0] * self.n_port  # Reset the action bitmap for the next step
        # state归一化并存储
        for port_idx in range(self.n_port):
            start_index = port_idx * self.nhp.port_states
            # NOTE: The k_min_norm, k_max_norm, and p_max has been always limited to [20, 50], [50, 100] and [0, 1] range in ns-3 side.
            #       So there is no need to adjust them even if the buffer size is not set as 400 KB in the ns-3 simulation. However, 
            #       the queue_length_norm should be adjusted according to the switch buffer size to avoid underestimation of queuing
            #       level and wrong reward calculation.
            port_obs = PortObservation(
                queue_length_norm=min(1, obs[start_index] * self.nhp.switch_buffer_size / 400),
                tx_rate_norm=obs[start_index + 1],
                ecn_rate_norm=obs[start_index + 2],
                k_min_norm=obs[start_index + 3],
                k_max_norm=obs[start_index + 4],
                p_max=obs[start_index + 5]
            )
            self.obs_history[port_idx].append(port_obs)
            logger.info(f"Step {curr_step} - Port {port_idx} - Observation {port_obs}.")

        logger.info(f"Step {curr_step} - Done {done}")
        return done
    

    def get_port_current_parameters(self, port_idx):
        curr_port_obs = self.obs_history[port_idx][-1]
        para = DCQCNParameters(k_min_norm=curr_port_obs.k_min_norm,
                              k_max_norm=curr_port_obs.k_max_norm,
                              p_max=curr_port_obs.p_max)
        return para


    def get_port_current_observation_list(self, port_idx):
        return self.obs_history[port_idx][-1].to_list()
    

    def get_state_dimension(self):
        return self.nhp.port_states * self.nhp.state_observations


    def get_port_last_state_list(self, port_idx):
        last_port_state_list = []
        for history_idx in [0, 1, 2]:
            last_port_state_list += self.obs_history[port_idx][history_idx].to_list()
        return last_port_state_list


    def get_port_current_state_list(self, port_idx):
        curr_port_state_list = []
        for history_idx in [1, 2, 3]:
            curr_port_state_list += self.obs_history[port_idx][history_idx].to_list()
        return curr_port_state_list
    
    # 端口reward计算
    def get_port_current_reward(self, port_idx):
        """
        Calculate the reward for the specified port based on the current observation.
        """
        def qlen_reward_mapping(queue_length_norm):
            """
            NOTE: In our setting, the maximum buffer size would be 1024 KB. So the qlen_reward will be zero if qlen > 1024 KB, even if 
                    it is very uncommon to have such a large queue length given the PFC exists.
            """
            import bisect
            qlen_reward = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0]
            qlen_norm_milestones = [2**i / 1024 for i in range(11)]

            return qlen_reward[bisect.bisect_left(qlen_norm_milestones, queue_length_norm)]

        curr_port_obs = self.obs_history[port_idx][-1]

        REWARD_WEIGHT_QLEN = 0.5
        reward = ((1-REWARD_WEIGHT_QLEN) * curr_port_obs.tx_rate_norm +
                  REWARD_WEIGHT_QLEN * qlen_reward_mapping(curr_port_obs.queue_length_norm))

        return reward
    

    def get_n_port(self):
        return self.n_port