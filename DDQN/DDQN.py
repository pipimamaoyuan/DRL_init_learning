import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gym
import random
from collections import deque
import matplotlib.pyplot as plt
import os

# 设置随机种子，保证结果可复现
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

MAX_Train_EPISODES = 1000  # 最大训练回合数
MAX_Train_STEPS = 500      # 每回合最大步数

MAX_Test_EPISODES = 10     # 最大测试回合数

class DQN(nn.Module):
    """
    定义深度Q网络（Deep Q-Network）
    这是一个简单的全连接神经网络，用于近似Q值函数
    """
    def __init__(self, state_size, action_size, hidden_size=64):
        """
        初始化神经网络
        Args:
            state_size: 状态空间的维度（输入层大小）
            action_size: 动作空间的维度（输出层大小）
            hidden_size: 隐藏层神经元数量
        """
        super(DQN, self).__init__()
        # 定义网络层
        self.fc1 = nn.Linear(state_size, hidden_size)   # 第一层全连接层
        self.fc2 = nn.Linear(hidden_size, hidden_size)  # 第二层全连接层
        self.fc3 = nn.Linear(hidden_size, action_size)  # 输出层，每个动作对应一个Q值
        
    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入状态
        Returns:
            所有动作的Q值
        """
        x = F.relu(self.fc1(x))  # 第一层 + ReLU激活函数
        x = F.relu(self.fc2(x))  # 第二层 + ReLU激活函数
        return self.fc3(x)       # 输出层，不需要激活函数

class ReplayBuffer:
    """
    经验回放缓冲区
    用于存储和采样智能体与环境交互的经验
    """
    def __init__(self, capacity):
        """
        初始化经验回放缓冲区
        Args:
            capacity: 缓冲区的最大容量
        """
        self.buffer = deque(maxlen=capacity)  # 使用双端队列，自动移除最老的样本(FIFO)
    
    def push(self, state, action, reward, next_state, done):
        """
        将经验存入缓冲区
        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一个状态
            done: 是否结束episode
        """
        # 将经验以元组形式存储
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """
        从缓冲区中随机采样一批经验
        Args:
            batch_size: 批量大小
        Returns:
            批量的状态、动作、奖励、下一个状态、完成标志
        """
        # 随机采样
        batch = random.sample(self.buffer, batch_size)  # 从缓冲区中随机采样
        
        # 将数据分别提取并转换为numpy数组
        states = np.array([experience[0] for experience in batch])
        actions = np.array([experience[1] for experience in batch])
        rewards = np.array([experience[2] for experience in batch])
        next_states = np.array([experience[3] for experience in batch])
        dones = np.array([experience[4] for experience in batch])
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        """返回缓冲区中当前的经验数量"""
        return len(self.buffer)

class DDQNAgent:
    """
    Double DQN智能体
    实现DDQN算法的核心逻辑
    """
    def __init__(self, state_size, action_size, lr=1e-3, gamma=0.99, 
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995,
                 buffer_size=10000, batch_size=64, target_update=100):
        """
        初始化DDQN智能体
        Args:
            state_size: 状态空间维度
            action_size: 动作空间维度
            lr: 学习率
            gamma: 折扣因子
            epsilon: 探索率（ε-greedy策略）
            epsilon_min: 最小探索率
            epsilon_decay: 探索率衰减率
            buffer_size: 经验回放缓冲区大小
            batch_size: 训练批量大小
            target_update: 目标网络更新频率
        """
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update  # 目标网络更新频率
        
        # 设备选择：如果有GPU则使用GPU，否则使用CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
        # 创建在线网络和目标网络
        self.q_network = DQN(state_size, action_size).to(self.device)
        self.target_network = DQN(state_size, action_size).to(self.device)
        
        # 复制在线网络的参数到目标网络
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # 目标网络设置为评估模式（不需要计算梯度）
        # 由于网络仅包含 Linear + ReLU，没有 Dropout、BatchNorm 等层，
        # 不调用 self.target_network.eval() 不会对训练结果或逻辑产生实质性影响。
        self.target_network.eval()
        
        # 优化器：用于更新在线网络的参数
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # 经验回放缓冲区
        self.memory = ReplayBuffer(buffer_size)
        
        # 训练步数计数器
        self.train_step = 0  # 更新（训练）网络的次数

    def act(self, state):
        """
        根据当前状态选择动作（ε-greedy策略）
        Args:
            state: 当前状态
        Returns:
            选择的动作
        """
        # 以epsilon的概率进行探索
        if np.random.random() <= self.epsilon:
            # 随机选择动作
            return random.randrange(self.action_size)
        else:
            # 利用策略：选择Q值最大的动作
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            # 在推理（inference）阶段（如选择动作）应始终使用 with torch.no_grad():，
            # 以避免不必要的梯度计算，节省内存和计算资源（虽然结果不会改变）
            with torch.no_grad():  # 禁用梯度计算
                q_values = self.q_network(state) # 计算每个动作的Q值
            return np.argmax(q_values.cpu().data.numpy()) # 返回Q值最大的动作
    
    def remember(self, state, action, reward, next_state, done):
        """
        将经验存储到经验回放缓冲区
        """
        self.memory.push(state, action, reward, next_state, done)
    
    def replay(self):
        """
        从经验回放缓冲区采样并训练网络
        Returns:
            损失值
        """
        # 如果缓冲区中的经验数量不足一个批次，则不进行训练
        if len(self.memory) < self.batch_size:
            return None
        
        # 从经验回放缓冲区采样一个批次的经验
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # 将numpy数组转换为PyTorch张量，并移动到相应设备
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # 获取当前状态-动作对的Q值
        # gather(1, actions.unsqueeze(1)) 用于选择对应动作的Q值
        # 当前 Q 值（预测值）
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # ========== DDQN的核心部分：计算目标Q值 ==========
        # 步骤1：使用在线网络选择下一个状态的最佳动作
        next_actions = self.q_network(next_states).max(1)[1].unsqueeze(1)
        
        # 步骤2：使用目标网络评估这些动作的Q值
        next_q_values = self.target_network(next_states).gather(1, next_actions).squeeze(1)
        
        # 计算目标Q值：如果episode结束，则只有即时奖励；否则加上折扣后的未来奖励
        # 目标 Q 值（通过 Bellman 方程计算）
        # 被视为当前对 q_values的“更优估计”（因为用了真实奖励rewards和未来估计next_q_values）
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # 计算损失：均方误差损失
        # 是为了让 Q 网络预测的 Q 值尽可能接近“目标 Q 值”（即通过 Bellman 方程计算出的更可靠估计）
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values.detach())
        
        # 反向传播和优化
        self.optimizer.zero_grad()  # 清空梯度
        loss.backward()             # 反向传播计算梯度
        self.optimizer.step()       # 更新网络参数
        
        # 衰减探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # 更新训练步数
        self.train_step += 1
        
        # 定期更新目标网络（硬更新）
        if self.train_step % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            
        # 硬更新（Hard Update）：
        # 每隔若干步，直接将主网络（online network）的参数完全复制给目标网络（target network）；
        
        # 软更新（Soft Update / Polyak Update）：
        # 每次训练都小幅更新目标网络参数，使其缓慢跟踪主网络
        
        return loss.item()
    
    def save(self, filepath):
        """
        保存模型参数
        Args:
            filepath: 保存路径
        """
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),            # 存储在线网络的参数
            'target_network_state_dict': self.target_network.state_dict(),  # 存储目标网络的参数
            'optimizer_state_dict': self.optimizer.state_dict(),            # 存储优化器的参数
            'epsilon': self.epsilon                                         # 存储当前的探索率
        }, filepath)
        print(f"模型已保存到: {filepath}")
    
    def load(self, filepath):
        """
        加载模型参数
        Args:
            filepath: 模型文件路径
        """
        if os.path.isfile(filepath):
            checkpoint = torch.load(filepath, map_location=self.device)         # 加载模型
            self.q_network.load_state_dict(checkpoint['q_network_state_dict'])  # 加载在线网络参数
            self.target_network.load_state_dict(checkpoint['target_network_state_dict'])    # 加载目标网络参数
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # 加载优化器参数
            self.epsilon = checkpoint['epsilon']                                # 加载探索率
            print(f"模型已从 {filepath} 加载")
        else:
            print(f"错误: 文件 {filepath} 不存在")

def train_ddqn(env, agent, episodes=MAX_Train_EPISODES, max_steps=MAX_Train_STEPS, save_path='DDQN/ddqn_model.pth'):
    """
    训练DDQN智能体
    Args:
        env: 环境
        agent: DDQN智能体
        episodes: 训练的总episode数
        max_steps: 每个episode的最大步数
        save_path: 模型保存路径
    Returns:
        每个episode的奖励和损失
    """
    scores = []          # 存储每个episode的总奖励
    losses = []          # 存储每个episode的平均损失
    recent_scores = deque(maxlen=100)  # 最近100个episode的奖励（用于计算平均）
    
    print("开始训练...")
    
    for episode in range(episodes):
        state, _ = env.reset()  # 重置环境，获取初始状态
        total_reward = 0
        episode_losses = []
        
        for steps in range(max_steps):
            # 选择动作
            action = agent.act(state)
            
            # 执行动作，获取环境反馈
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # 将经验存储到经验回放缓冲区
            agent.remember(state, action, reward, next_state, done)
            
            # 训练网络
            loss = agent.replay()
            if loss is not None:
                episode_losses.append(loss)
            
            # 更新状态和总奖励
            state = next_state
            total_reward += reward
            
            # 如果episode结束，跳出循环
            if done:
                break
        
        # 记录结果
        scores.append(total_reward)
        recent_scores.append(total_reward)
        
        # 计算平均损失
        avg_loss = np.mean(episode_losses) if episode_losses else 0
        losses.append(avg_loss)
        
        # 计算最近100个episode的平均奖励
        mean_recent_score = np.mean(recent_scores) if recent_scores else total_reward
        
        # 每100个episode打印一次训练进度
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode+1}/{episodes}, "
                  f"Score: {total_reward}, "
                  f"Average Score (last 100): {mean_recent_score:.2f}, "
                  f"Epsilon: {agent.epsilon:.3f}, "
                  f"Average Loss: {avg_loss:.4f}")
        
        # 如果最近100个episode的平均奖励达到目标，提前结束训练
        if mean_recent_score >= 195.0 and len(recent_scores) == 100:
            print(f"问题已解决！在 {episode+1} 个episode后达到目标。")
            break
    
    # 保存训练好的模型
    agent.save(save_path)
    
    return scores, losses

def test_ddqn(env, agent, episodes=MAX_Test_EPISODES, render=True):
    """
    测试训练好的DDQN智能体
    Args:
        env: 环境
        agent: 训练好的DDQN智能体
        episodes: 测试的episode数
        render: 是否渲染环境（显示动画）
    """
    print("开始测试...")
    test_scores = []
    
    for episode in range(episodes):
        state, _ = env.reset() # 重置环境，获取初始状态
        total_reward = 0
        steps = 0
        
        while True:
            if render:
                env.render()
            
            # 使用训练好的策略选择动作（不探索）
            action = agent.act(state)
            
            # 执行动作
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            total_reward += reward
            steps += 1
            state = next_state
            
            if done:
                break
        
        test_scores.append(total_reward)
        print(f"Test Episode {episode+1}: Score = {total_reward}, Steps = {steps}")
    
    env.close()
    
    avg_score = np.mean(test_scores)
    print(f"平均测试分数: {avg_score:.2f}")
    return test_scores

def plot_results(scores, losses, window=100):
    """
    可视化训练结果
    Args:
        scores: 每个episode的奖励
        losses: 每个episode的损失
        window: 滑动平均的窗口大小
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 绘制奖励曲线
    ax1.plot(scores, alpha=0.6, label='Reward')
    
    # 计算滑动平均
    if len(scores) >= window:
        moving_avg = [np.mean(scores[i-window:i]) for i in range(window, len(scores))]
        ax1.plot(range(window, len(scores)), moving_avg, label=f'{window} Episode Moving Average', color='red', linewidth=2)

    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Training Process - Reward')
    ax1.legend()
    ax1.grid(True)
    
    # 绘制损失曲线
    ax2.plot(losses, alpha=0.6, color='orange')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Loss')
    ax2.set_title('Training Process - Loss')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('DDQN/ddqn_training_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """
    主函数：完整的训练和测试流程
    """
    # 创建环境
    env = gym.make('CartPole-v1')
    
    # 设置随机种子
    env.reset(seed=SEED)
    
    # 获取状态和动作空间的维度
    state_size = env.observation_space.shape[0]  # CartPole-v1的状态维度为4
    action_size = env.action_space.n             # CartPole-v1的动作维度为2（左或右）
    
    print(f"环境: CartPole-v1")
    print(f"状态空间维度: {state_size}")
    print(f"动作空间维度: {action_size}")
    
    # 创建DDQN智能体
    agent = DDQNAgent(
        state_size=state_size,   # 状态空间维度
        action_size=action_size, # 动作空间维度
        lr=1e-3,           # 学习率
        gamma=0.99,        # 折扣因子
        epsilon=1.0,       # 初始探索率
        epsilon_min=0.01,  # 最小探索率
        epsilon_decay=0.995, # 探索率衰减
        buffer_size=10000, # 经验回放缓冲区大小
        batch_size=32,     # 批量大小
        target_update=100  # 目标网络更新频率
    )
    
    # 训练模型
    scores, losses = train_ddqn(env, agent, episodes=MAX_Train_EPISODES, max_steps=MAX_Train_STEPS, save_path='DDQN/ddqn_cartpole.pth')
    
    # 可视化训练结果
    plot_results(scores, losses)
    
    # 重新加载训练好的模型进行测试
    trained_agent = DDQNAgent(state_size, action_size)
    trained_agent.load('DDQN/ddqn_cartpole.pth')
    
    # 测试模型（不显示动画）
    test_scores = test_ddqn(env, trained_agent, episodes=10, render=False)
    
    # 如果需要观看智能体的表现，取消下面这行的注释
    # test_ddqn(env, trained_agent, episodes=3, render=True)
    
    env.close()

if __name__ == "__main__":
    main()