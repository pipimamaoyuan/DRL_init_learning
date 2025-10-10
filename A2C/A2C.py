import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import time
import os

# 设备检测（有可用 CUDA 则使用 GPU）
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 设置随机种子，保证结果可复现
torch.manual_seed(42)
np.random.seed(42)

MAX_Train_EPISODES = 5000
MAX_Train_STEPS = 2000

MAX_TEST_EPISODES = 10
MAX_TEST_STEPS = 2000

DEMO_EPISODES_number = 3  # 演示时的回合数

class ActorCriticNetwork(nn.Module):
    """
    Actor-Critic网络：共享特征提取层，分别输出策略和价值
    - Actor网络：输出动作的概率分布
    - Critic网络：输出状态的价值估计
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(ActorCriticNetwork, self).__init__()
        
        # 共享的特征提取层
        self.shared_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor网络：输出动作的概率分布
        self.actor = nn.Linear(hidden_dim, action_dim)
        
        # Critic网络：输出状态价值
        self.critic = nn.Linear(hidden_dim, 1)
        
    def forward(self, state):
        """
        前向传播
        输入：状态 state
        输出：动作概率分布 action_probs, 状态价值 state_value
        """
        # 提取共享特征
        features = self.shared_net(state)
        
        # Actor分支：通过softmax得到动作概率分布
        action_logits = self.actor(features)
        action_probs = F.softmax(action_logits, dim=-1)
        
        # Critic分支：输出状态价值
        state_value = self.critic(features)
        
        return action_probs, state_value

class A2CAgent:
    """
    A2C智能体类：包含训练和测试的核心逻辑
    """
    def __init__(self, state_dim, action_dim, learning_rate=0.001, gamma=0.99, entropy_coef=0.01, device=None):
        # 超参数设置
        self.state_dim = state_dim   # 状态维度
        self.action_dim = action_dim # 动作维度
        self.learning_rate = learning_rate   # 学习率
        self.gamma = gamma   # 折扣因子
        self.entropy_coef = entropy_coef   # 熵正则化系数
        # 选择设备（优先使用传入device，其次为全局DEVICE）
        self.device = torch.device(device) if device is not None else DEVICE

        # 创建Actor-Critic网络并移动到device
        self.network = ActorCriticNetwork(state_dim, action_dim).to(self.device)

        # 优化器（在模型移动到device之后创建）
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        
        # 用于记录训练过程
        self.episode_rewards = []
        self.losses = []
        
    def select_action(self, state):
        """
        根据当前状态选择动作
        输入：状态 state (numpy array)
        输出：动作 action, 动作的对数概率 log_prob, 状态价值 value
        """
        # 将numpy数组转换为torch tensor，并添加batch维度，放到agent所在device
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        # 通过网络获取动作概率和状态价值
        # NOTE: Do not disable gradients here so that log_probs and values
        # keep their computation graph for later backprop during update.
        action_probs, state_value = self.network(state_tensor)

        # 创建分类分布，用于采样动作
        distribution = Categorical(action_probs)

        # 从分布中采样一个动作
        action = distribution.sample()

        # 计算该动作的对数概率
        log_prob = distribution.log_prob(action)

        # 返回动作值（去掉batch维度），对数概率，状态价值（去掉batch维度），以及动作概率分布
        return action.item(), log_prob, state_value.squeeze(), action_probs

    def update(self, rewards, log_probs, values, action_probs_list, next_value, done):
        """
        A2C算法的核心更新步骤
        输入：
            rewards: 奖励序列
            log_probs: 动作对数概率序列  
            values: 状态价值序列
            next_value: 下一个状态的价值
            done: 是否终止
        """
        # 计算优势函数和回报
        advantages = []
        returns = []

        # 如果episode终止，下一个状态的价值为0，否则使用估计值
        R = 0 if done else next_value.detach()

        # 反向计算累积回报和优势函数
        for reward in reversed(rewards):
            R = reward + self.gamma * R
            returns.insert(0, R)

        # 将列表转换为tensor（确保dtype与网络输出一致，便于做反向传播）
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        log_probs = torch.stack(log_probs).to(self.device) 
        values = torch.stack(values).squeeze().to(self.device)
        action_probs = torch.stack(action_probs_list)  # 形状: [time_steps, action_dim]
        
        # 计算优势函数：A(s,a) = R - V(s)
        advantages = returns - values
        
        # 计算策略损失：-log_prob * advantage
        policy_loss = -(log_probs * advantages.detach()).mean()
        
        # 计算价值损失：(R - V(s))^2
        value_loss = advantages.pow(2).mean()
        
        # 计算策略的熵（用于鼓励探索）
        # 正确的熵计算方法：-sum(p * log(p))
        # 使用完整的动作概率分布计算熵: -sum(p * log(p))
        entropy = - (action_probs * torch.log(action_probs + 1e-8)).sum(dim=-1)  # 对动作维度求和
        entropy_loss = entropy.mean()  # 对时间步求平均
        
        # 总损失 = 策略损失 + 价值损失 - 熵正则项
        total_loss = policy_loss + 0.5 * value_loss - self.entropy_coef * entropy_loss
        
        # 反向传播和优化
        self.optimizer.zero_grad()
        total_loss.backward()

        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)

        self.optimizer.step()
        
        # 记录损失
        self.losses.append(total_loss.item())
        
        return total_loss.item()

def train_a2c(env_name='CartPole-v1', num_episodes=MAX_Train_EPISODES, max_steps=MAX_Train_STEPS):
    """
    训练A2C智能体的主函数
    """
    # 创建环境
    env = gym.make(env_name) 
    
    # 获取状态和动作维度
    state_dim = env.observation_space.shape[0]  # 获取状态维度
    action_dim = env.action_space.n  # 获取动作维度
    
    print(f"环境: {env_name}")
    print(f"状态维度: {state_dim}, 动作维度: {action_dim}")
    
    # 创建A2C智能体
    agent = A2CAgent(state_dim, action_dim)
    
    print("开始训练...")
    
    # 训练循环
    for episode in range(num_episodes):
        state = env.reset()  # 重置环境，获取初始状态
        if isinstance(state, tuple):
            state = state[0]  # 新版本gym返回tuple
        
        episode_reward = 0
        rewards = []
        log_probs = []          # 保存动作对数概率
        values = []             # 保存状态价值
        action_probs_list = []  # 保存动作概率分布
        
        # 单个episode的交互循环
        for step in range(max_steps):
            # 选择动作
            # 返回动作值（去掉batch维度），对数概率，状态价值（去掉batch维度），以及动作概率分布
            action, log_prob, value, action_probs = agent.select_action(state) 
            
            # 执行动作
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # 记录数据
            rewards.append(reward)
            log_probs.append(log_prob)  # 保存动作对数概率
            values.append(value)        # 状态价值
            action_probs_list.append(action_probs.squeeze(0))  # 保存动作概率分布
            
            episode_reward += reward
            state = next_state
            
            # 如果episode结束或者达到最大步数，进行更新
            if done or step == max_steps - 1:
                # 计算下一个状态的价值（如果episode结束则为0）
                if done:
                    next_value = torch.tensor(0.0, device=agent.device)
                else:
                    next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(agent.device)
                    with torch.no_grad():
                        _, next_value = agent.network(next_state_tensor)
                        next_value = next_value.squeeze()
                
                # 更新网络参数
                # 输入：奖励序列，动作对数概率序列，状态价值序列，动作概率分布，下一状态价值，是否结束标志
                agent.update(rewards, log_probs, values, action_probs_list, next_value, done)
                break
        
        # 记录每个episode的奖励
        agent.episode_rewards.append(episode_reward)
        
        # 每100个episode打印一次训练进度
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(agent.episode_rewards[-100:])
            print(f"Episode {episode+1}, 平均奖励 (最近100轮): {avg_reward:.2f}")
            
            # 如果平均奖励达到环境的最大值，提前结束训练
            if avg_reward >= max_steps:
                print(f"训练完成！在 {episode+1} 个episode后达到最大奖励。")
                break
    
    # 关闭环境
    env.close()
    
    return agent

def save_model(agent, file_path='A2C/a2c_cartpole.pth'):
    """
    保存训练好的模型
    """
    # 确保目录存在
    os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else '.', exist_ok=True)
    
    # 保存模型参数
    torch.save({
        'network_state_dict': agent.network.state_dict(), # 保存网络参数
        'optimizer_state_dict': agent.optimizer.state_dict(), # 优化器参数
        'episode_rewards': agent.episode_rewards, # 训练过程中每个episode的奖励
        'losses': agent.losses # 训练过程中每个更新步骤的损失
    }, file_path)
    
    print(f"模型已保存到: {file_path}")

def load_model(file_path='A2C/a2c_cartpole.pth', env_name='CartPole-v1'):
    """
    加载训练好的模型
    """
    # 创建环境和智能体
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0] # 获取状态维度
    action_dim = env.action_space.n # 获取动作维度
    agent = A2CAgent(state_dim, action_dim) # 创建智能体实例

    # 加载模型参数（确保根据可用device映射）
    map_location = agent.device
    checkpoint = torch.load(file_path, map_location=map_location) # 加载模型
    agent.network.load_state_dict(checkpoint['network_state_dict']) # 加载网络参数
    agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict']) # 加载优化器参数
    agent.episode_rewards = checkpoint['episode_rewards'] # 加载每个episode的奖励
    agent.losses = checkpoint['losses'] # 加载每个更新步骤的损失

    env.close()
    
    print(f"模型已从 {file_path} 加载")
    return agent

def test_agent(agent, env_name='CartPole-v1', num_episodes=10, render=True):
    """
    测试训练好的智能体
    """
    env = gym.make(env_name)
    
    test_rewards = []
    
    for episode in range(num_episodes):
        state = env.reset()  # 重置环境，获取初始状态
        if isinstance(state, tuple):
            state = state[0]
        
        episode_reward = 0
        done = False
        
        while not done:
            if render:
                env.render()
                time.sleep(0.02)  # 减慢速度以便观察
            
            # 选择动作（只使用网络预测，不计算梯度）
            with torch.no_grad():
                action, _, _, _ = agent.select_action(state)
            
            # 执行动作
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            state = next_state
        
        test_rewards.append(episode_reward)
        print(f"测试 Episode {episode+1}: 奖励 = {episode_reward}")
    
    env.close()
    
    avg_reward = np.mean(test_rewards)
    print(f"\n平均测试奖励: {avg_reward:.2f}")
    
    return test_rewards

def plot_training_progress(agent):
    """
    可视化训练过程
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 绘制奖励曲线
    ax1.plot(agent.episode_rewards)
    ax1.set_title('Reward of Training')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.grid(True)
    
    # 计算滑动平均奖励（窗口大小为100）
    if len(agent.episode_rewards) >= 100:
        moving_avg = np.convolve(agent.episode_rewards, np.ones(100)/100, mode='valid')
        ax1.plot(range(99, len(agent.episode_rewards)), moving_avg, 'r-', linewidth=2, label='Average Reward (100)')
        ax1.legend()
    
    # 绘制损失曲线
    ax2.plot(agent.losses)
    ax2.set_title('Loss During Training')
    ax2.set_xlabel('Update Steps')
    ax2.set_ylabel('Loss')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('A2C/a2c_training_progress.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """
    主函数：完整的训练、保存、测试、可视化流程
    """
    print("=" * 50)
    print("A2C算法求解 CartPole-v1")
    print("=" * 50)
    
    # 1. 训练A2C智能体
    print("\n1. 训练阶段")
    agent = train_a2c(num_episodes=MAX_Train_STEPS)
    
    # 2. 保存训练好的模型
    print("\n2. 保存模型")
    save_model(agent, 'A2C/a2c_cartpole.pth')
    
    # 3. 可视化训练过程
    print("\n3. 可视化训练过程")
    plot_training_progress(agent)
    
    # 4. 测试训练好的智能体
    print("\n4. 测试阶段")
    test_rewards = test_agent(agent, num_episodes=MAX_TEST_EPISODES, render=True)
    print(f"\n平均测试奖励: {np.mean(test_rewards):.2f}")
    
    # 5. 演示加载模型并测试
    print("\n5. 演示模型加载")
    loaded_agent = load_model('A2C/a2c_cartpole.pth')
    test_agent(loaded_agent, num_episodes=DEMO_EPISODES_number, render=True)
    
    print("\n程序执行完成！")

if __name__ == "__main__":
    main()