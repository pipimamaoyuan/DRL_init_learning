import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import random
from collections import deque
import matplotlib.pyplot as plt
import os

MAX_Train_EPISODES = 1000  # 训练的最大回合数
MAX_Test_EPISODES = 10    # 测试的最大回合数

# 设置随机种子以保证结果可复现
def set_seed(seed, env=None):
    """
    设定随机种子以保证结果可复现。
    如果提供了 env（Gymnasium 环境），使用 env.reset(seed=seed) 和 action_space/observation_space 的 seed 方法。
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if env is not None:
        # Gymnasium 推荐在 reset 时传入 seed。
        try:
            env.reset(seed=seed)
        except Exception:
            # 兼容老版本 gym 的 env.seed
            try:
                env.seed(seed)
            except Exception:
                pass

        # 同步 action/observation 空间的随机种子（如果支持）
        try:
            env.action_space.seed(seed)
        except Exception:
            pass
        try:
            env.observation_space.seed(seed)
        except Exception:
            pass

# 定义Q网络（Critic网络）
class QNetwork(nn.Module):
    """
    Q网络：
    输入: 状态 和 动作;
    输出: 对应的Q值
    在SAC中我们使用两个Q网络（Q1和Q2）来避免Q值的高估
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(QNetwork, self).__init__()
        # 第一层：状态和动作拼接后作为输入
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)  # 输出单个Q值
        
        # 初始化权重
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        
    def forward(self, state, action):
        # 将状态和动作在特征维度上拼接
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value

# 定义策略网络（Actor网络）
class PolicyNetwork(nn.Module):
    """
    策略网络：
    输入：状态，
    输出：动作的均值 和 标准差
    在SAC中策略是随机策略，输出动作的分布参数
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256, log_std_min=-20, log_std_max=2):
        super(PolicyNetwork, self).__init__()
        self.log_std_min = log_std_min # 最小allowed std
        self.log_std_max = log_std_max # 最大allowed std
        
        # 共享的特征提取层
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # 输出均值的层
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        # 输出对数标准差的层
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)
        
        # 初始化权重
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.mean_layer.weight)
        nn.init.xavier_uniform_(self.log_std_layer.weight)
        
    def forward(self, state):
        # 特征提取
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        # 计算均值和标准差
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        
        # 限制标准差的范围，保证数值稳定性
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        
        return mean, std
    
    def sample(self, state, epsilon=1e-6):
        """
        从策略分布中采样动作，使用重参数化技巧保证梯度可传
        """
        mean, std = self.forward(state) # 获取动作的均值和标准差
        
        # 从标准正态分布中采样噪声
        noise = torch.randn_like(mean)  # 生成标准正态分布的噪声
        
        # 重参数化技巧：使用均值和标准差变换噪声
        # 这样梯度可以通过这个变换回传到网络参数
        action = mean + std * noise
        
        # 计算动作的对数概率（用于策略梯度更新）
        log_prob = self.log_prob(mean, std, noise)
        
        # 使用tanh将动作限制在[-1, 1]范围内，并计算对应的对数概率修正
        action_tanh = torch.tanh(action)
        log_prob -= torch.log(1 - action_tanh.pow(2) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action_tanh, log_prob
    
    def log_prob(self, mean, std, noise):
        """
        计算给定噪声下动作的对数概率
        """
        # 高斯分布的对数概率密度函数
        return (-0.5 * noise.pow(2) - torch.log(std) - 0.5 * np.log(2 * np.pi)).sum(1, keepdim=True)

# 定义值函数网络（Value Network）
class ValueNetwork(nn.Module):
    """
    值函数网络：输入状态，输出状态值V(s)
    在SAC中用于稳定训练，有些实现会省略这个网络
    """
    def __init__(self, state_dim, hidden_dim=256):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
        # 初始化权重
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)
        return value

# SAC算法主类
class SAC:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2, automatic_entropy_tuning=True):
        """
        SAC算法初始化
        参数说明：
        - state_dim: 状态维度
        - action_dim: 动作维度  
        - lr: 学习率
        - gamma: 折扣因子
        - tau: 目标网络软更新系数
        - alpha: 熵权重系数
        - automatic_entropy_tuning: 是否自动调整熵系数
        """
        self.gamma = gamma # 折扣因子
        self.tau = tau  # 软更新参数
        self.alpha = alpha # 熵系数
        self.automatic_entropy_tuning = automatic_entropy_tuning # 是否自动调整熵系数
        
        # 设备选择：如果有GPU则使用GPU，否则使用CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
        # 初始化策略网络（Actor）
        self.policy_net = PolicyNetwork(state_dim, action_dim).to(self.device)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        
        # 初始化两个Q网络（Critic），用于减少过高估计
        self.q_net1 = QNetwork(state_dim, action_dim).to(self.device)
        self.q_net2 = QNetwork(state_dim, action_dim).to(self.device)
        self.q_optimizer1 = optim.Adam(self.q_net1.parameters(), lr=lr)
        self.q_optimizer2 = optim.Adam(self.q_net2.parameters(), lr=lr)
        
        # 新增：Value Network及其目标网络
        self.value_net = ValueNetwork(state_dim).to(self.device)
        self.target_value_net = ValueNetwork(state_dim).to(self.device)
        
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr)
        self.target_value_net.load_state_dict(self.value_net.state_dict()) # 将目标值网络初始化为当前值网络
        
        # 初始化目标Q网络
        self.target_q_net1 = QNetwork(state_dim, action_dim).to(self.device)
        self.target_q_net2 = QNetwork(state_dim, action_dim).to(self.device)
       
        # 将Q网络的权重复制到目标Q网络
        self.target_q_net1.load_state_dict(self.q_net1.state_dict())
        self.target_q_net2.load_state_dict(self.q_net2.state_dict())
        
        # 如果启用自动熵调整，则初始化熵系数为可学习参数
        if self.automatic_entropy_tuning:
            self.target_entropy = -torch.prod(torch.Tensor([action_dim]).to(self.device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
        
        # 经验回放缓冲区
        self.replay_buffer = deque(maxlen=100000) # 最大容量10万
        
    def select_action(self, state, evaluate=False):
        """
        根据状态选择动作
        - evaluate: 是否为评估模式，在评估模式下不使用随机性
        """
        # 有些 Gymnasium 版本的 env.reset()/env.step() 返回 (obs, info)
        # 这里确保 state 为单个 observation（ndarray / list），而不是包含 info 的元组
        if isinstance(state, (tuple, list)):
            # 常见情况：state == (obs, info)
            state = state[0]

        # 确保 state 是一个数值序列（例如 numpy array 或 list），再转换为 tensor
        state = torch.FloatTensor(np.array(state)).unsqueeze(0).to(self.device)
        
        if evaluate:
            # 评估模式：直接使用均值作为动作（确定性策略）
            with torch.no_grad():
                mean, _ = self.policy_net(state) # 获取动作的均值
                action = torch.tanh(mean)
        else:
            # 训练模式：从分布中采样动作（随机策略）
            with torch.no_grad():
                action, _ = self.policy_net.sample(state)
        
        return action.cpu().numpy()[0]
    
    def update(self, batch_size):
        """
        使用一批经验数据更新网络参数
        """
        if len(self.replay_buffer) < batch_size:
            return 0, 0, 0
        
        # 从经验回放缓冲区中随机采样一批数据
        batch = random.sample(self.replay_buffer, batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
        
        # 将数据转换为PyTorch张量
        # 为避免从 numpy.ndarray 列表直接创建 tensor 导致的性能问题，
        # 先将列表转换为单个 numpy.ndarray，然后再使用 torch.from_numpy。
        state_np = np.array(state_batch, dtype=np.float32)
        action_np = np.array(action_batch, dtype=np.float32)
        reward_np = np.array(reward_batch, dtype=np.float32)
        next_state_np = np.array(next_state_batch, dtype=np.float32)
        done_np = np.array(done_batch, dtype=np.float32)

        state_batch = torch.from_numpy(state_np).to(self.device)
        action_batch = torch.from_numpy(action_np).to(self.device)
        reward_batch = torch.from_numpy(reward_np).unsqueeze(1).to(self.device)
        next_state_batch = torch.from_numpy(next_state_np).to(self.device)
        done_batch = torch.from_numpy(done_np).unsqueeze(1).to(self.device)
        
        # 1. 更新 value 网络
        with torch.no_grad():
            # 从策略网络中采样下一个动作和对应的对数概率
            next_action, next_log_prob = self.policy_net.sample(next_state_batch)
            
            # 计算目标Q值（使用两个目标Q网络的最小值来减少过高估计）
            target_q1 = self.target_q_net1(next_state_batch, next_action)
            target_q2 = self.target_q_net2(next_state_batch, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            # 计算目标Value：Q值减去熵项
            target_value = target_q - self.alpha * next_log_prob
        
        # 当前Value网络的预测
        current_value = self.value_net(state_batch)
        # 计算Value网络的损失（均方误差）
        value_loss = F.mse_loss(current_value, target_value)
        # 更新Value网络
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()
        
        # 2. 更新Q Network
        with torch.no_grad():
            # 使用目标Value网络计算下一个状态的值
            next_value = self.target_value_net(next_state_batch)
            # 目标Q值：奖励 + 折扣因子 * 下一个状态的值
            target_q_value = reward_batch + (1 - done_batch) * self.gamma * next_value
        
        # 计算当前Q网络的预测值
        current_q1 = self.q_net1(state_batch, action_batch)
        current_q2 = self.q_net2(state_batch, action_batch)
        
        # 计算Q网络的损失（均方误差）
        q1_loss = F.mse_loss(current_q1, target_q_value)
        q2_loss = F.mse_loss(current_q2, target_q_value)
        
        # 更新Q网络
        self.q_optimizer1.zero_grad()
        q1_loss.backward()
        self.q_optimizer1.step()
        
        self.q_optimizer2.zero_grad()
        q2_loss.backward()
        self.q_optimizer2.step()
        
        # 3. 更新策略网络
        # 重新采样动作以计算策略梯度（因为策略已经更新）
        new_action, log_prob = self.policy_net.sample(state_batch)
        
        # 计算策略损失：最小化 (alpha * log_prob - Q_value)
        q1_new = self.q_net1(state_batch, new_action)
        q2_new = self.q_net2(state_batch, new_action)
        q_new = torch.min(q1_new, q2_new)
        
        # 策略损失：alpha * log_prob - Q_value
        policy_loss = (self.alpha * log_prob - q_new).mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        # 如果启用自动熵调整，则更新熵系数alpha
        alpha_loss = None
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha.exp()
        
        # 软更新所有目标网络
        self.soft_update(self.q_net1, self.target_q_net1)
        self.soft_update(self.q_net2, self.target_q_net2)
        self.soft_update(self.value_net, self.target_value_net)
        
        # 返回Q损失（取两个Q网络损失的平均值）、策略损失、Value损失和alpha损失
        avg_q_loss = (q1_loss.item() + q2_loss.item()) / 2

        return avg_q_loss, policy_loss.item(), value_loss.item(), alpha_loss.item() if alpha_loss is not None else 0

    def soft_update(self, source_net, target_net):
        """
        软更新目标网络参数：θ_target = τ * θ_source + (1 - τ) * θ_target
        """
        for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
            target_param.data.copy_(self.tau * source_param.data + (1 - self.tau) * target_param.data)
    
    def save_model(self, path):
        """
        保存模型参数
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)  # 创建目录
        torch.save({
            'policy_net': self.policy_net.state_dict(),  # 策略网络
            'q_net1': self.q_net1.state_dict(),          #  Q网络1
            'q_net2': self.q_net2.state_dict(),          #  Q网络2
            'value_net': self.value_net.state_dict(),  # 新增 值函数网络
            'target_q_net1': self.target_q_net1.state_dict(), # 目标Q网络1
            'target_q_net2': self.target_q_net2.state_dict(), # 目标Q网络2
            'target_value_net': self.target_value_net.state_dict(),  # 新增 目标值函数网络
        }, path)
        print(f"模型已保存到: {path}")
    
    def load_model(self, path):
        """
        加载模型参数
        """
        checkpoint = torch.load(path, map_location=self.device)   # 加载模型
        self.policy_net.load_state_dict(checkpoint['policy_net']) # 策略网络
        self.q_net1.load_state_dict(checkpoint['q_net1'])       # Q网络1
        self.q_net2.load_state_dict(checkpoint['q_net2'])       # Q网络2
        self.value_net.load_state_dict(checkpoint['value_net']) # 新增 值函数网络
        self.target_q_net1.load_state_dict(checkpoint['target_q_net1']) # 目标Q网络1
        self.target_q_net2.load_state_dict(checkpoint['target_q_net2']) # 目标Q网络2
        self.target_value_net.load_state_dict(checkpoint['target_value_net']) # 新增 目标值函数网络
        print(f"模型已从 {path} 加载")

# 训练函数
def train_sac(env, agent, episodes=MAX_Train_EPISODES, batch_size=256, update_interval=1, save_interval=100):
    """
    训练SAC代理
    """
    rewards = []  # 记录每个episode的总奖励
    q_losses = []  # 记录Q网络损失
    policy_losses = []  # 记录策略网络损失
    value_losses = []  # 新增：记录Value网络损失
    alpha_losses = []  # 记录熵系数损失
    
    print("开始训练...")
    
    for episode in range(episodes):
        # Gymnasium: env.reset() 返回 (obs, info)
        state, _ = env.reset() # 重置环境，获取初始状态
        episode_reward = 0 # 记录episode的总奖励
        steps = 0          # 记录总步数
        episode_q_loss = 0 # 存储Q网络损失
        episode_policy_loss = 0 # 存储策略网络损失
        episode_value_loss = 0 # 新增 存储值函数网络损失
        episode_alpha_loss = 0  # 存储熵系数损失
        update_count = 0    # 存储更新次数
        
        while True:
            # 选择动作
            action = agent.select_action(state)
            
            # 执行动作
            # Gymnasium: env.step() 返回 (obs, reward, terminated, truncated, info)
            # SAC 策略网络输出的动作通常被限制在 [-1, 1] 范围内，
            # 但Pendulum-v1的动作范围是[-2, 2]
            next_state, reward, terminated, truncated, _ = env.step(action * 2) 
            done = terminated or truncated
            
            # 存储经验到回放缓冲区
            agent.replay_buffer.append((state, action, reward, next_state, done))
            
            # 更新网络
            if len(agent.replay_buffer) > batch_size and steps % update_interval == 0:
                q_loss, policy_loss, value_loss, alpha_loss = agent.update(batch_size) # 更新网络
                episode_q_loss += q_loss # 累加Q网络损失
                episode_policy_loss += policy_loss # 累加策略网络损失
                episode_value_loss += value_loss # 累加值函数网络损失
                episode_alpha_loss += alpha_loss   # 累加熵系数损失
                update_count += 1 # 累加更新次数
            
            state = next_state
            episode_reward += reward
            steps += 1
            
            if done:
                break
        
        # 计算平均损失
        if update_count > 0:
            avg_q_loss = episode_q_loss / update_count
            avg_policy_loss = episode_policy_loss / update_count
            avg_value_loss = episode_value_loss / update_count # 新增 计算平均值函数网络损失
            avg_alpha_loss = episode_alpha_loss / update_count
        else:
            avg_q_loss = avg_policy_loss = avg_value_loss = avg_alpha_loss = 0
            
        rewards.append(episode_reward)
        q_losses.append(avg_q_loss)
        policy_losses.append(avg_policy_loss)
        value_losses.append(avg_value_loss)  # 新增：记录值函数网络损失
        alpha_losses.append(avg_alpha_loss)
        
        # 打印训练进度
        if (episode + 1) % 5 == 0:
            print(f"Episode: {episode+1}, Reward: {episode_reward:.2f}, Steps: {steps}, "
                  f"Q Loss: {avg_q_loss:.4f}, Policy Loss: {avg_policy_loss:.4f}, "
                  f"Value Loss: {avg_value_loss:.4f}, Alpha: {agent.alpha.item():.4f}")

        # 定期保存模型
        if (episode + 1) % save_interval == 0:
            agent.save_model(f"SAC_with_value_network/sac_pendulum_episode_{episode+1}.pth")
    
    # 训练完成后保存最终模型
    agent.save_model("SAC_with_value_network/sac_pendulum_final.pth")

    return rewards, q_losses, policy_losses, value_losses, alpha_losses

# 测试函数
def test_sac(env, agent, model_path, test_episodes=MAX_Test_EPISODES, render=True):
    """
    测试训练好的SAC代理
    """
    # 加载模型
    agent.load_model(model_path)
    
    print("开始测试...")
    test_rewards = []
    
    for episode in range(test_episodes):
        state, _ = env.reset() # 重置环境
        episode_reward = 0
        
        while True:
            # 在测试时使用确定性策略
            action = agent.select_action(state, evaluate=True)
            
            # 执行动作
            next_state, reward, terminated, truncated, _ = env.step(action * 2)
            done = terminated or truncated
            
            if render:
                env.render()
            
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        test_rewards.append(episode_reward)
        print(f"Test Episode: {episode+1}, Reward: {episode_reward:.2f}")
    
    avg_reward = np.mean(test_rewards)
    print(f"平均测试奖励: {avg_reward:.2f}")
    
    return test_rewards

# 可视化训练结果
def plot_training_results(rewards, q_losses, policy_losses, value_losses, alpha_losses, window=10):
    """
    绘制训练结果图表
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 平滑奖励曲线
    if len(rewards) >= window:
        smoothed_rewards = [np.mean(rewards[i-window:i]) for i in range(window, len(rewards))]
    else:
        smoothed_rewards = rewards
    
    # 绘制奖励
    ax1.plot(rewards, alpha=0.3, label='Raw Reward')
    ax1.plot(range(window, len(rewards)), smoothed_rewards, label=f'Smoothed (Window={window})')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Training Reward')
    ax1.legend()
    ax1.grid(True)
    
    # 绘制Q网络损失
    ax2.plot(q_losses)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Loss of Q')
    ax2.set_title('Loss of Q')
    ax2.grid(True)
    
    # 绘制策略网络损失
    ax3.plot(policy_losses)
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Loss of Policy')
    ax3.set_title('Loss of Policy')
    ax3.grid(True)
    
    # 绘制熵系数损失
    ax4.plot(alpha_losses)
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Loss of Alpha')
    ax4.set_title('Loss of Alpha')
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig('SAC_with_value_network/training_results_1.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    fig,ax5 = plt.subplots(1, 1, figsize=(10, 5))
    ax5.plot(value_losses)
    ax5.set_xlabel('Episode')
    ax5.set_ylabel('Loss of Value')
    ax5.set_title('Loss of Value')
    ax5.grid(True)
    
    plt.tight_layout()
    plt.savefig('SAC_with_value_network/training_results_2.png', dpi=300, bbox_inches='tight')
    plt.show()

# 主程序
if __name__ == "__main__":
    # 创建环境
    env = gym.make('Pendulum-v1')
    
    # 设置随机种子（同时传入 env 以设置环境的种子）
    set_seed(42, env)
    
    # 获取状态和动作维度
    state_dim = env.observation_space.shape[0]  # Pendulum-v1的状态维度为3
    action_dim = env.action_space.shape[0]  # Pendulum-v1的动作维度为1
    
    print(f"状态维度: {state_dim}, 动作维度: {action_dim}")
    
    # 创建SAC代理
    agent = SAC(
        state_dim=state_dim,  # 状态维度
        action_dim=action_dim,  # 动作维度
        lr=3e-4,  # 学习率
        gamma=0.99,  # 折扣因子
        tau=0.005,   # soft更新参数
        automatic_entropy_tuning=True  # 是否自动调整熵系数
    )
    
    # 训练参数
    episodes = MAX_Train_EPISODES  # 训练回合数
    batch_size = 256  # 批次大小
    update_interval = 1  # 更新间隔
    
    # 开始训练
    rewards, q_losses, policy_losses, value_losses, alpha_losses = train_sac(
        env, agent, episodes, batch_size, update_interval
    )
    
    # 绘制训练结果
    plot_training_results(rewards, q_losses, policy_losses, value_losses, alpha_losses)
    
    # 测试训练好的模型
    print("\n开始测试最终模型...")
    test_rewards = test_sac(env, agent, "SAC_with_value_network/sac_pendulum_final.pth", test_episodes=MAX_Test_EPISODES, render=True)
    
    # 关闭环境
    env.close()
    
    print("程序执行完毕！")