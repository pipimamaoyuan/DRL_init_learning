import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque
import matplotlib.pyplot as plt
import os

# 设置随机种子，保证结果可复现
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

MAX_Train_EPISODES_NUMBER = 1000  # 最大训练回合数
MAX_Test_EPISODES_NUMBER = 10          # 最大测试回合数

# 1. 定义Q网络（神经网络部分）
class QNetwork(nn.Module):
    """
    Q网络：输入状态，输出每个动作的Q值
    CartPole-v1的状态空间：4维向量 [车的位置, 车的速度, 杆的角度, 杆的角速度]
    动作空间：2个动作 [向左推, 向右推]
    """
    def __init__(self, state_size, action_size, hidden_size=64):
        super(QNetwork, self).__init__()
        # 定义神经网络层
        self.fc1 = nn.Linear(state_size, hidden_size)  # 第一层全连接层
        self.fc2 = nn.Linear(hidden_size, hidden_size) # 第二层全连接层
        self.fc3 = nn.Linear(hidden_size, action_size) # 输出层，输出每个动作的Q值
        
    def forward(self, state):
        """
        前向传播：输入状态，输出Q值
        """
        x = F.relu(self.fc1(state))  # 第一层 + ReLU激活函数
        x = F.relu(self.fc2(x))      # 第二层 + ReLU激活函数
        q_values = self.fc3(x)       # 输出层，不需要激活函数
        return q_values

# 2. 定义经验回放缓冲区
class ReplayBuffer:
    """
    经验回放缓冲区：存储智能体的经验 (state, action, reward, next_state, done)
    通过随机采样打破数据间的相关性，提高训练稳定性
    """
    def __init__(self, buffer_size, batch_size, device=DEVICE):
        self.memory = deque(maxlen=buffer_size)  # 使用双端队列存储经验
        self.batch_size = batch_size             # 每次训练的批次大小
        self.device = device
        
    def add(self, state, action, reward, next_state, done):
        """
        添加一条经验到缓冲区
        """
        experience = (state, action, reward, next_state, done)
        self.memory.append(experience)
        
    def sample(self):
        """
        从缓冲区中随机采样一个批次的经验
        """
        # 如果经验数量不足批次大小，则返回空
        if len(self.memory) < self.batch_size:
            return None
        
        batch = random.sample(self.memory, self.batch_size) # 随机采样

        # 将经验分解为单独的数组
        # 为了避免从 numpy.ndarray 列表直接创建 tensor 导致的性能问题，
        # 先将列表转换为单个 numpy.ndarray，然后再使用 torch.from_numpy。
        states_np = np.array([exp[0] for exp in batch], dtype=np.float32)
        actions_np = np.array([exp[1] for exp in batch], dtype=np.int64)
        rewards_np = np.array([exp[2] for exp in batch], dtype=np.float32)
        next_states_np = np.array([exp[3] for exp in batch], dtype=np.float32)
        dones_np = np.array([exp[4] for exp in batch], dtype=np.bool_)

        # 将numpy数组转换为tensor并移动到指定device
        states = torch.from_numpy(states_np).float().to(self.device)
        actions = torch.from_numpy(actions_np).long().to(self.device)
        rewards = torch.from_numpy(rewards_np).float().to(self.device)
        next_states = torch.from_numpy(next_states_np).float().to(self.device)
        dones = torch.from_numpy(dones_np).to(self.device)

        return (states, actions, rewards, next_states, dones)
    
    def __len__(self):
        """
        返回当前缓冲区中的经验数量
        """
        return len(self.memory)

# 3. 定义DQN智能体
class DQNAgent:
    """
    DQN智能体：包含Q网络、目标网络、经验回放缓冲区等核心组件
    """
    def __init__(self, state_size, action_size, device=DEVICE):
        self.state_size = state_size # 状态空间大小
        self.action_size = action_size # 动作空间大小
        self.device = device
        
        # Q网络和目标网络
        # 初始化网络并移动到device
        self.q_network = QNetwork(state_size, action_size).to(self.device)
        self.target_network = QNetwork(state_size, action_size).to(self.device)
        
        # 优化器：用于更新Q网络的参数
        # 目标网络（target_network）的作用是提供一个“稳定的目标”来训练主网络。
        # 如果我们也直接优化它，就会失去这个稳定性，导致训练发散。
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)
        
        # 经验回放缓冲区，创建时传入device以便直接生成在device上的张量
        self.memory = ReplayBuffer(buffer_size=10000, batch_size=32, device=self.device)
        
        # 超参数
        self.gamma = 0.99          # 折扣因子，衡量未来奖励的重要性
        self.epsilon = 1.0         # 探索率，初始为1.0（完全随机探索）
        self.epsilon_min = 0.01    # 最小探索率
        self.epsilon_decay = 0.995 # 探索率衰减率
        self.update_target_every = 100  # 每多少步更新一次目标网络
        
        # 训练步数计数器
        self.train_step = 0
        
    def act(self, state):
        """
        根据当前状态选择动作
        使用epsilon-greedy策略：以epsilon的概率随机探索，以1-epsilon的概率选择最优动作
        """
        # 随机探索
        if np.random.random() <= self.epsilon:
            return random.choice(range(self.action_size))
        
        # 利用：选择Q值最大的动作
        else:
            # 将numpy数组转换为torch张量，并添加batch维度，移动到device
            state_tensor = torch.from_numpy(np.array(state, dtype=np.float32)).unsqueeze(0).to(self.device)
            # 使用Q网络计算Q值（不计算梯度，因为只是推理）
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            # 返回Q值最大的动作（取回cpu上的标量）
            return q_values.argmax().item()
    
    def remember(self, state, action, reward, next_state, done):
        """
        将经验存储到回放缓冲区
        """
        self.memory.add(state, action, reward, next_state, done)
    
    def replay(self):
        """
        从经验回放缓冲区中采样并训练Q网络
        """
        # 采样一个批次的经验
        batch = self.memory.sample()
        if batch is None:
            return
            
        states, actions, rewards, next_states, dones = batch

        # 计算当前Q值：Q_network(states)得到所有动作的Q值，然后gather选择执行的动作对应的Q值
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # 计算目标Q值
        with torch.no_grad():
            # 使用目标网络计算下一个状态的最大Q值
            next_q_values = self.target_network(next_states).max(1)[0]
            # dones 可能是bool类型，转为 float：done=1 -> 1.0 表示结束
            not_done = (1.0 - dones.float())
            # 如果回合结束，目标Q值就是当前奖励；否则是当前奖励加上折扣后的未来奖励
            target_q_values = rewards + (self.gamma * next_q_values * not_done)
        
        # 计算损失：均方误差损失
        loss = F.mse_loss(current_q_values, target_q_values)
        
        # 梯度清零、反向传播、更新参数
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 衰减探索率，但不能低于最小值
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        # 更新目标网络（定期硬更新）
        self.train_step += 1
        if self.train_step % self.update_target_every == 0:
            self.update_target_network()
    
    def update_target_network(self):
        """
        将Q网络的参数复制到目标网络
        """
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def save(self, filepath):
        """
        保存模型参数
        """
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(), # 主网络参数
            'target_network_state_dict': self.target_network.state_dict(), # 目标网络参数
            'optimizer_state_dict': self.optimizer.state_dict(), # 优化器参数
            'epsilon': self.epsilon, # 探索率
            'train_step': self.train_step # 训练步数
        }, filepath)
    
    def load(self, filepath):
        """
        加载模型参数
        """
        # 使用 map_location 确保模型加载到当前 agent 的 device 上
        checkpoint = torch.load(filepath, map_location=self.device) # 加载模型参数
        self.q_network.load_state_dict(checkpoint['q_network_state_dict']) # 加载主网络参数
        self.target_network.load_state_dict(checkpoint['target_network_state_dict']) # 加载目标网络参数
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict']) # 加载优化器参数
        self.epsilon = checkpoint['epsilon'] # 加载探索率
        self.train_step = checkpoint['train_step'] # 加载训练步数

# 4. 训练函数
def train_dqn(episodes, render=False, save_path='dqn_cartpole.pth'):
    """
    训练DQN智能体
    """
    # 创建环境和智能体
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0] # 状态空间维度
    action_size = env.action_space.n            # 动作空间维度
    agent = DQNAgent(state_size, action_size)
    
    # 记录每个回合的得分
    scores = []
    # 记录最近100个回合的平均得分（用于判断是否解决环境）
    recent_scores = deque(maxlen=100) # 使用deque实现队列
    
    print("开始训练DQN...")
    
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        
        # 一个回合的循环
        while not done:
            if render and episode % 100 == 0:  # 每100回合渲染一次
                env.render()
                
            # 选择动作
            action = agent.act(state)
            # 执行动作，获取下一个状态和奖励
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # 存储经验
            agent.remember(state, action, reward, next_state, done)
            
            # 更新状态和总奖励
            state = next_state
            total_reward += reward
            
            # 训练智能体
            agent.replay()
        
        # 记录得分
        scores.append(total_reward)
        recent_scores.append(total_reward)
        
        # 计算最近100回合的平均得分
        mean_score = np.mean(recent_scores)
        
        # 每50回合打印一次训练进度
        if episode % 50 == 0:
            print(f"Episode: {episode}, Score: {total_reward}, Epsilon: {agent.epsilon:.3f}, Mean Score: {mean_score:.2f}")
        
        # 如果最近100回合平均得分>=195，认为问题已解决
        if mean_score >= 195.0 and len(recent_scores) >= 100:
            print(f"环境已在第 {episode} 回合解决！平均得分: {mean_score:.2f}")
            break
    
    # 保存训练好的模型
    agent.save(save_path)
    print(f"模型已保存到: {save_path}")
    
    env.close()
    return scores, agent

# 5. 测试函数
def test_dqn(model_path, test_episodes=10, render=True):
    """
    测试训练好的DQN模型
    """
    # 创建环境和智能体
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0] # 状态空间维度
    action_size = env.action_space.n            # 动作空间维度
    agent = DQNAgent(state_size, action_size)   # 创建DQN智能体
    
    # 加载训练好的模型
    agent.load(model_path)
    
    # 设置探索率为0，完全利用
    agent.epsilon = 0.0
    
    print("开始测试DQN模型...")
    
    test_scores = []
    
    for episode in range(test_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            if render:
                env.render()
                
            # 选择动作（完全利用，不探索）
            action = agent.act(state)
            # 执行动作
            next_state, reward, done, truncated, _ = env.step(action)
            done = done or truncated
            
            state = next_state
            total_reward += reward
        
        test_scores.append(total_reward)
        print(f"Test Episode {episode+1}: Score = {total_reward}")
    
    env.close() # 关闭环境
    
    mean_score = np.mean(test_scores)
    std_score = np.std(test_scores)
    print(f"Test Results: Mean Score = {mean_score:.2f} ± {std_score:.2f}")
    
    return test_scores

# 6. 可视化训练结果
def plot_results(scores, window=100):
    """
    绘制训练得分曲线
    """
    plt.figure(figsize=(12, 6))
    
    # 绘制原始得分
    plt.subplot(1, 2, 1)
    plt.plot(scores, alpha=0.6, label='Raw Score')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('DQN Training Progress - Raw Score')
    plt.legend()
    plt.grid(True)
    
    # 绘制移动平均得分
    plt.subplot(1, 2, 2)
    # 计算移动平均
    moving_avg = []
    for i in range(len(scores)):
        if i < window:
            moving_avg.append(np.mean(scores[:i+1]))
        else:
            moving_avg.append(np.mean(scores[i-window+1:i+1]))
    
    plt.plot(moving_avg, color='red', linewidth=2, label=f'{window} Episode Moving Average')
    plt.axhline(y=195, color='green', linestyle='--', label='Solved Threshold (195)')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('DQN Training Progress - Moving Average')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('DQN/dqn_training_results.png', dpi=300, bbox_inches='tight')
    plt.show()

# 7. 主程序
if __name__ == "__main__":
    # 设置模型保存路径
    MODEL_PATH = "DQN/dqn_cartpole.pth"
    
    # 检查是否有已训练的模型
    if os.path.exists(MODEL_PATH):
        print("发现已训练的模型，是否重新训练？(y/n)")
        choice = input().strip().lower()
        if choice == 'y':
            # 训练模型
            scores, agent = train_dqn(episodes=MAX_Train_EPISODES_NUMBER, render=False, save_path=MODEL_PATH)
            # 绘制训练结果
            plot_results(scores)
        else:
            print("使用已有模型进行测试...")
    else:
        # 训练模型
        scores, agent = train_dqn(episodes=MAX_Train_EPISODES_NUMBER, render=False, save_path=MODEL_PATH)
        # 绘制训练结果
        plot_results(scores)
    
    # 测试模型
    print("\n" + "="*50)
    test_scores = test_dqn(MODEL_PATH, test_episodes=MAX_Test_EPISODES_NUMBER, render=True)
    
    # 显示最终测试结果
    print("\n" + "="*50)
    print("DQN算法在CartPole-v1环境上的表现总结:")
    print(f"- 最终测试平均得分: {np.mean(test_scores):.2f}")
    print(f"- 最佳测试得分: {np.max(test_scores)}")
    print(f"- 最差测试得分: {np.min(test_scores)}")
    print(f"- 稳定性(标准差): {np.std(test_scores):.2f}")
    
    # 判断是否成功解决问题
    if np.mean(test_scores) >= 195:
        print("✅ DQN成功解决了CartPole-v1环境！")
    else:
        print("❌ DQN未能完全解决CartPole-v1环境，可能需要更多训练。")