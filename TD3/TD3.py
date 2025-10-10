import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import collections
import random
import matplotlib.pyplot as plt
import os
import time

# 设置随机种子，保证结果可复现
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# 设备选择：优先使用GPU（CUDA），否则使用CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

MAX_Train_EPISODES = 1000
MAX_Train_STEPS = 200

MAX_TEST_EPISODES = 10
MAX_TEST_STEPS = 200

DEMO_EPISODES_number = 3  # 演示时的回合数
DEMO_STEPS_number = 200  # 演示时每回合的最大步数


# 兼容gym和gymnasium的reset函数
def reset_env(env, seed=None):
    """
    Reset environment and return the observation only.
    Works with both gym (returns obs) and gymnasium (returns (obs, info)).
    If seed is provided, pass it to env.reset(seed=seed).
    """
    if seed is not None:
        ret = env.reset(seed=seed)
    else:
        ret = env.reset()

    # gymnasium returns (obs, info), gym returns obs
    if isinstance(ret, tuple) or isinstance(ret, list):
        return ret[0]
    return ret

# 兼容gym和gymnasium的step函数
def step_env(env, action):
    """
    Step environment and return (obs, reward, done, info) compatible with older gym.
    Handles both gymnasium (obs, reward, terminated, truncated, info) and
    classic gym (obs, reward, done, info).
    """
    ret = env.step(action)
    # gymnasium: (obs, reward, terminated, truncated, info)
    if isinstance(ret, tuple) and len(ret) == 5:
        obs, reward, terminated, truncated, info = ret
        done = bool(terminated or truncated)
        return obs, reward, done, info
    # classic gym: (obs, reward, done, info)
    if isinstance(ret, tuple) and len(ret) == 4:
        obs, reward, done, info = ret
        return obs, reward, bool(done), info
    # Fallback: try to unpack first 4
    try:
        obs, reward, done, info = ret
        return obs, reward, bool(done), info
    except Exception:
        # unexpected format
        return ret, 0.0, True, {}

class Actor(nn.Module):
    """
    Actor网络（策略网络）
    输入：状态state
    输出：确定性动作action（在Pendulum环境中，动作是连续的，范围[-2, 2]）
    """
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        
        # 定义网络层结构
        self.layer1 = nn.Linear(state_dim, 400)  # 第一层：400个神经元
        self.layer2 = nn.Linear(400, 300)        # 第二层：300个神经元
        self.layer3 = nn.Linear(300, action_dim) # 输出层：维度等于动作维度
        
        self.max_action = max_action  # 动作的最大值，用于缩放输出
        
    def forward(self, state):
        """
        前向传播
        state: 输入状态
        返回：确定性动作
        """
        # 使用ReLU激活函数，最后一层使用tanh将输出限制在[-1, 1]范围内
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))
        # 使用tanh将输出限制在[-1, 1]，然后乘以max_action缩放到环境需要的范围
        action = self.max_action * torch.tanh(self.layer3(x))
        return action

class Critic(nn.Module):
    """
    Critic网络（价值网络）- 双Q网络
    输入：状态state和动作action
    输出：Q值，表示在给定状态下执行给定动作的价值
    TD3使用两个独立的Critic网络来缓解Q值过估计问题
    """
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        
        # 第一个Q网络
        self.layer1_q1 = nn.Linear(state_dim + action_dim, 400)
        self.layer2_q1 = nn.Linear(400, 300)
        self.layer3_q1 = nn.Linear(300, 1)  # 输出单个Q值
        
        # 第二个Q网络（独立参数）
        self.layer1_q2 = nn.Linear(state_dim + action_dim, 400)
        self.layer2_q2 = nn.Linear(400, 300)
        self.layer3_q2 = nn.Linear(300, 1)
        
    def forward(self, state, action):
        """
        前向传播，返回两个Q值
        """
        # 将状态和动作拼接在一起作为输入
        state_action = torch.cat([state, action], 1)
        
        # 第一个Q网络的前向传播
        q1 = F.relu(self.layer1_q1(state_action))
        q1 = F.relu(self.layer2_q1(q1))
        q1 = self.layer3_q1(q1)
        
        # 第二个Q网络的前向传播
        q2 = F.relu(self.layer1_q2(state_action))
        q2 = F.relu(self.layer2_q2(q2))
        q2 = self.layer3_q2(q2)
        
        return q1, q2
    
    def q1(self, state, action):
        """
        只返回第一个Q网络的Q值，用于策略更新
        """
        state_action = torch.cat([state, action], 1)
        q1 = F.relu(self.layer1_q1(state_action))
        q1 = F.relu(self.layer2_q1(q1))
        q1 = self.layer3_q1(q1)
        return q1

class ReplayBuffer:
    """
    经验回放缓冲区
    用于存储和采样智能体与环境交互的经验（状态、动作、奖励、下一个状态、是否终止）
    """
    def __init__(self, max_size):
        self.buffer = collections.deque(maxlen=max_size)  # 使用双端队列，自动限制最大长度
    
    def add(self, state, action, reward, next_state, done):
        """
        添加一条经验到缓冲区
        """
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        """
        从缓冲区中随机采样一个批次的经验
        """
        batch = random.sample(self.buffer, batch_size) # 从缓冲区中随机采样一个batch_size的样本
        
        # 将经验拆分成不同的数组
        states = np.array([experience[0] for experience in batch])
        actions = np.array([experience[1] for experience in batch])
        rewards = np.array([experience[2] for experience in batch])
        next_states = np.array([experience[3] for experience in batch])
        dones = np.array([experience[4] for experience in batch])
        
        # 转换为PyTorch张量
        # 转换为PyTorch张量并移动到设备（GPU/CPU）
        states = torch.FloatTensor(states).to(DEVICE)
        actions = torch.FloatTensor(actions).to(DEVICE)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(DEVICE)  # 添加维度，从[batch]变为[batch, 1]
        next_states = torch.FloatTensor(next_states).to(DEVICE)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(DEVICE)

        return states, actions, rewards, next_states, dones
    
    def size(self):
        """
        返回当前缓冲区中的经验数量
        """
        return len(self.buffer)

class TD3:
    """
    TD3算法主类
    """
    def __init__(self, state_dim, action_dim, max_action):
        # 超参数设置
        self.batch_size = 64           # 批次大小
        self.gamma = 0.99              # 折扣因子
        self.tau = 0.005               # 目标网络软更新参数
        self.policy_noise = 0.2        # 目标策略平滑的噪声标准差
        self.noise_clip = 0.5          # 目标策略平滑的噪声裁剪范围
        self.policy_freq = 2           # 策略网络更新频率（延迟更新）
        
        self.max_action = max_action   # 动作最大值，用于裁剪动作
        self.total_it = 0              # 总迭代次数
        
    # 创建Actor和Critic网络
        # 创建并移动到设备
        self.actor = Actor(state_dim, action_dim, max_action).to(DEVICE)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(DEVICE)  # Actor目标网络
        
        self.actor_target.load_state_dict(self.actor.state_dict())    # 初始化目标网络参数
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim).to(DEVICE)
        self.critic_target = Critic(state_dim, action_dim).to(DEVICE)            # Critic目标网络
        
        self.critic_target.load_state_dict(self.critic.state_dict())  # 初始化目标网络参数
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=3e-4)
        
        # 创建经验回放缓冲区
        self.replay_buffer = ReplayBuffer(1000000)  # 缓冲区容量为100万
        
    def select_action(self, state, add_noise=True):
        """
        根据当前状态选择动作
        add_noise: 是否在动作上添加探索噪声
        """
        # 将状态转换为张量并移动到设备
        state = torch.FloatTensor(state.reshape(1, -1)).to(DEVICE)
        with torch.no_grad():
            action = self.actor(state).cpu().data.numpy().flatten()  # 通过Actor网络得到动作
        
        if add_noise:
            # 添加探索噪声（高斯噪声）
            noise = np.random.normal(0, 0.1, size=action.shape)
            action = (action + noise).clip(-self.max_action, self.max_action)  # 限制动作在合法范围内
        
        return action
    
    def train(self):
        """
        训练一步：从经验回放缓冲区采样并更新网络参数
        """
        self.total_it += 1  # 网络总训练（更新）次数加1
        
        # 从经验回放缓冲区采样一个批次
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # -------------------- 更新Critic网络 -------------------- #
        with torch.no_grad():  # 在计算目标值时不需要梯度
            # TD3核心技巧1：目标策略平滑
            # 在目标策略的动作上添加噪声，然后裁剪到合法范围
            noise = (
                torch.randn_like(actions) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)
            
            # 通过目标Actor网络得到下一个动作，并添加噪声
            next_actions = (
                self.actor_target(next_states) + noise
            ).clamp(-self.max_action, self.max_action)
            
            # TD3核心技巧2：双Q网络
            # 计算两个目标Q值，并取最小值来缓解Q值过估计
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            
            # 计算目标Q值：r + γ * (1 - done) * Q_target
            target_q = rewards + self.gamma * (1 - dones) * target_q
        
        # 获取当前Q值估计
        current_q1, current_q2 = self.critic(states, actions)
        
        # 计算Critic损失（均方误差）
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        # 更新Critic网络
        self.critic_optimizer.zero_grad()  # 清空梯度
        critic_loss.backward()             # 反向传播
        self.critic_optimizer.step()       # 更新参数
        
        # -------------------- 更新Actor网络（延迟更新） -------------------- #
        # TD3核心技巧3：延迟策略更新
        # 每隔policy_freq步才更新一次Actor网络和目标网络
        if self.total_it % self.policy_freq == 0:
            # 替换 actor_loss 计算
            q1, q2 = self.critic(states, self.actor(states))
            actor_loss = -torch.min(q1, q2).mean()
            
            # 更新Actor网络
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # -------------------- 软更新目标网络 -------------------- #
            # 使用软更新方式更新目标网络参数，而不是直接复制
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def save(self, filename):
        """
        保存模型参数
        """
        torch.save({
            'actor_state_dict': self.actor.state_dict(),            # 保存actor网络参数
            'critic_state_dict': self.critic.state_dict(),          # 保存critic网络参数
            'actor_target_state_dict': self.actor_target.state_dict(),  # 保存actor_target网络参数
            'critic_target_state_dict': self.critic_target.state_dict(), # 保存critic_target网络参数
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(), # 保存actor_optimizer参数
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(), # 保存critic_optimizer参数
        }, filename)
    
    def load(self, filename):
        """
        加载模型参数
        """
        # 使用 map_location 将模型加载到当前设备（GPU或CPU）
        checkpoint = torch.load(filename, map_location=DEVICE)          # 加载模型参数
        self.actor.load_state_dict(checkpoint['actor_state_dict'])      # 加载actor网络参数
        self.critic.load_state_dict(checkpoint['critic_state_dict'])    # 加载critic网络参数
        self.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])  # 加载actor_target网络参数
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict']) # 加载critic_target网络参数
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict']) # 加载actor_optimizer参数
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict']) # 加载critic_optimizer参数

        # 确保模型在目标设备上
        self.actor.to(DEVICE)
        self.actor_target.to(DEVICE)
        self.critic.to(DEVICE)
        self.critic_target.to(DEVICE)

        # 将优化器内部的张量移动到目标设备（如果有）
        for optim in (self.actor_optimizer, self.critic_optimizer):
            for state in optim.state.values():
                for k, v in list(state.items()):
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(DEVICE)

def train_td3(env_name="Pendulum-v1", max_episodes=MAX_Train_EPISODES, max_steps=MAX_Train_STEPS):
    """
    训练TD3算法
    """
    # 创建环境
    env = gym.make(env_name)

    # 获取环境信息
    state_dim = env.observation_space.shape[0]  # 状态维度
    action_dim = env.action_space.shape[0]      # 动作维度
    max_action = float(env.action_space.high[0])  # 动作最大值
    
    print(f"环境: {env_name}")
    print(f"状态维度: {state_dim}, 动作维度: {action_dim}, 最大动作值: {max_action}")
    
    # 创建TD3智能体
    agent = TD3(state_dim, action_dim, max_action)
    
    # 用于记录训练过程中的奖励
    rewards = []
    avg_rewards = []  # 平均奖励（滑动平均）
    
    print("开始训练...")
    
    # 训练循环
    for episode in range(max_episodes):
        # 兼容gym和gymnasium的reset函数
        if episode == 0:
            state = reset_env(env, seed=SEED)
        else:
            state = reset_env(env)
        episode_reward = 0
        
        for step in range(max_steps):
            # 选择动作（训练时添加探索噪声）
            action = agent.select_action(state, add_noise=True)
            
            # 执行动作
            next_state, reward, done, _ = step_env(env, action)
            
            # 将经验存储到回放缓冲区
            agent.replay_buffer.add(state, action, reward, next_state, done)
            
            # 更新状态和累计奖励
            state = next_state
            episode_reward += reward
            
            # 当缓冲区中有足够多的经验时开始训练
            if agent.replay_buffer.size() > 1000:
                agent.train()
            
            if done:
                break
        
        # 记录奖励
        rewards.append(episode_reward)
        
        # 计算滑动平均奖励（最近100个episode的平均）
        avg_reward = np.mean(rewards[-100:])
        avg_rewards.append(avg_reward)

        # 每1个episode打印一次训练信息
        if episode % 1 == 0:
            print(f"Episode: {episode}, Reward: {episode_reward:.2f}, Average Reward: {avg_reward:.2f}")
        
        # 如果平均奖励达到阈值，提前结束训练
        if avg_reward >= -200:
            print(f"训练完成！在Episode {episode}达到目标性能")
            break
    
    # 关闭环境
    env.close()
    
    # 保存训练好的模型
    agent.save("TD3/td3_pendulum.pth")
    print("模型已保存为 'TD3/td3_pendulum.pth'")

    return rewards, avg_rewards, agent

def test_td3(env_name="Pendulum-v1", model_path="TD3/td3_pendulum.pth", num_episodes=MAX_TEST_EPISODES):
    """
    测试训练好的TD3模型
    """
    # 创建环境
    env = gym.make(env_name)

    # 获取环境信息
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    # 创建TD3智能体并加载训练好的模型
    agent = TD3(state_dim, action_dim, max_action)
    agent.load(model_path)
    
    print("开始测试...")
    
    test_rewards = []
    
    for episode in range(num_episodes):
        state = reset_env(env)
        episode_reward = 0
        frames = []  # 用于存储每一帧（可视化用）

        for step in range(MAX_TEST_STEPS):  # Pendulum环境的最大步数是200
            # 选择动作（测试时不添加探索噪声）
            action = agent.select_action(state, add_noise=False)
            
            # 执行动作
            next_state, reward, done, _ = step_env(env, action)
            
            # 更新状态和累计奖励
            state = next_state
            episode_reward += reward
            
            # 如果需要可视化，可以保存当前帧
            # frame = env.render(mode='rgb_array')
            # frames.append(frame)
            
            if done:
                break
        
        test_rewards.append(episode_reward)
        print(f"测试Episode {episode+1}: 奖励 = {episode_reward:.2f}")
    
    env.close()
    
    avg_test_reward = np.mean(test_rewards)
    print(f"平均测试奖励: {avg_test_reward:.2f}")
    
    return test_rewards

def plot_training_curve(rewards, avg_rewards):
    """
    绘制训练曲线
    """
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(rewards, alpha=0.6, label='Reward')
    plt.plot(avg_rewards, label='Moving average reward (100 episodes)', color='orange')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Pendulum-v1 Training Progress with TD3')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    # 绘制最近100个回合的奖励分布
    if len(rewards) > 100:
        recent_rewards = rewards[-100:]
    else:
        recent_rewards = rewards
    
    plt.hist(recent_rewards, bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel('Reward')
    plt.ylabel('Frequency')
    plt.title('Reward distribution for the last 100 rounds')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('TD3/td3_training_curve.png', dpi=300, bbox_inches='tight')
    plt.show()

def demo_agent(env_name="Pendulum-v1", model_path="TD3/td3_pendulum.pth", num_episodes=DEMO_EPISODES_number):
    """
    演示训练好的智能体在环境中的表现
    """
    # 创建环境
    env = gym.make(env_name)

    # 获取环境信息
    state_dim = env.observation_space.shape[0]    # 状态维度
    action_dim = env.action_space.shape[0]        # 动作维度
    max_action = float(env.action_space.high[0])  # 最大动作值
    
    # 创建TD3智能体并加载训练好的模型
    agent = TD3(state_dim, action_dim, max_action)
    agent.load(model_path)
    
    print("开始演示...")
    
    for episode in range(num_episodes):
        state = reset_env(env) # 重置环境
        episode_reward = 0

        for step in range(DEMO_STEPS_number):
            # 渲染环境
            env.render()
            
            # 选择动作（测试时不添加探索噪声）
            action = agent.select_action(state, add_noise=False)
            
            # 执行动作
            next_state, reward, done, _ = step_env(env, action)
            
            # 更新状态和累计奖励
            state = next_state
            episode_reward += reward
            
            # 添加小延迟，方便观察
            time.sleep(0.02)
            
            if done:
                break
        
        print(f"演示Episode {episode+1}: 奖励 = {episode_reward:.2f}")
    
    env.close()

if __name__ == "__main__":
    # 训练TD3算法
    print("=== TD3算法训练 ===")
    rewards, avg_rewards, trained_agent = train_td3()
    
    # 绘制训练曲线
    print("\n=== 绘制训练曲线 ===")
    plot_training_curve(rewards, avg_rewards)
    
    # 测试训练好的模型
    print("\n=== 测试训练好的模型 ===")
    test_rewards = test_td3()
    
    # 演示智能体表现（可选，需要图形界面）
    # print("\n=== 演示智能体表现 ===")
    # demo_agent()
    
    print("\n=== 程序完成 ===")